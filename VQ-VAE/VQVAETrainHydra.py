import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

from VQVAEModel_positional import positional_VQVAE, get_time_embedding
from dataloader import *

from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
import os

import torchvision.transforms as T
from torchvision.utils import make_grid

from accelerate import DistributedDataParallelKwargs
import math
from time import time
import numpy as np

from ai4ha.log.textlog import textlog

from ai4ha.util import (
    instantiate_from_config,
    time_management,
    fix_paths,
    experiment_name_autoencoder,
    save_config,
)
from ai4ha.util.train import (
    get_most_recent_checkpoint,
    save_checkpoint_accelerate,
    get_optimizer,
    get_lr_scheduler,
)

import accelerate
import timm
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm.auto import trange
import matplotlib.pyplot as plt
import random

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")


def visualize_reconstructions(real_samples, reconstructed_samples, iteration, BASE_DIR, num_samples=4):
    # Choose a subset of samples to visualize
    real_samples = real_samples[:num_samples].detach().cpu()
    reconstructed_samples = reconstructed_samples[:num_samples].detach().cpu()
    path = f"{BASE_DIR}samples/E{iteration:04d}-S{iteration + 1:05d}s.png"
    # Plot the real and reconstructed samples
    if real_samples.shape[0] < num_samples:
        num_samples = real_samples.shape[0]
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
    for i in range(num_samples):
        # Real samples (top row)
        axes[0, i].imshow(real_samples[i].permute(1, 2, 0),
                          cmap=plt.get_cmap('gray'))  # If image data, permute for channels last
        axes[0, i].axis('off')
        axes[0, i].set_title("Real")

        # Reconstructed samples (bottom row)
        axes[1, i].imshow(reconstructed_samples[i].permute(1, 2, 0), cmap=plt.get_cmap('gray'))  # Same here
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    fig.savefig(path)


def train(model, train_loader, image_key, BASE_DIR, num_epochs=10, alpha=10, device='cuda', opt=None, num_codes=256,
          visualize_freq=10):
    # Initialize the Accelerator
    accelerator = Accelerator()  # Handles device placement and mixed precision
    accelerator.wait_for_everyone()
    device = accelerator.device  # Use the device determined by accelerator

    # Prepare the model, optimizer, and dataloader with accelerator
    model, opt, train_loader = accelerator.prepare(model, opt, train_loader)

    # Progress bar for tracking iterations
    best_loss = 1000
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader),
                            disable=not accelerator.is_local_main_process)
        training_data = {
            "rec_loss": [],
            "cmt_loss": []
        }
        for i, batch in enumerate(train_loader):

            pixel_values = (batch[image_key].permute(0, 3, 1, 2).to(
                memory_format=torch.contiguous_format).to(accelerator.device))
            positions = (batch["position"].to(
                memory_format=torch.contiguous_format).to(accelerator.device))
            pos_emb = get_time_embedding(positions, temb_dim=190)

            with accelerator.accumulate(model):
                out, encoder_out, indices, cmt_loss = model(pixel_values, pos_emb)   # Forward pass through the model
                rec_loss = (out - pixel_values).abs().mean()   # Reconstruction loss
                cmt_loss_scalar = cmt_loss.mean()
                # Backpropagation (wrapped by accelerator)
                accelerator.backward(rec_loss + alpha * cmt_loss_scalar)

                opt.step()  # Update model parameters
                opt.zero_grad()
                progress_bar.set_description(
                    f"Epoch {epoch} | " + f"rec loss: {rec_loss.item():.3f} | "
                    + f"cmt loss: {cmt_loss_scalar.item():.3f} | "
                    + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
                )
                training_data['rec_loss'].append(rec_loss.item())
                training_data['cmt_loss'].append(cmt_loss_scalar.item())
                
        if accelerator.is_main_process:
            print(f"Epoch: {epoch} | Reconstruction loss: {np.mean(training_data['rec_loss']):.3f} |  Cmt loss: {np.mean(training_data['cmt_loss']):.3f}")

            if epoch % visualize_freq == 0:
                real_samples, reconstructed_samples = accelerator.gather(pixel_values), accelerator.gather(out)
                visualize_reconstructions(real_samples, reconstructed_samples, epoch, BASE_DIR)

            if np.mean(training_data['rec_loss']) < best_loss:
                best_loss = np.mean(training_data['rec_loss'])
                torch.save(model, f'{BASE_DIR}model/VAE-best-model-epoch.pt')
                torch.save(model.state_dict(), f'{BASE_DIR}model/VAE-best-model-parameters-epoch.pt')
                best_model_data = {
                    "Epoch": epoch,
                    "Reconstruction loss": np.mean(training_data['rec_loss']),
                    "Cmt loss": np.mean(training_data['cmt_loss'])
                }

    print("End of Training")
    print(f"BEST MODEL INFO  |  Reconstruction loss: {best_model_data['Reconstruction loss']:.3f}  |  Epoch: {best_model_data['Epoch']}")


def main(config):
    BASE_DIR = f"{config['base_dir_logs']}{config['name']}/"
    print(BASE_DIR)
    os.makedirs(f"{BASE_DIR}", exist_ok=True)
    DIRS = ["checkpoints", "logs", "samples", "final", "model"]
    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    image_key = ("image" if "image_key" not in config["model"] else
                 config["model"]["image_key"])

    train_data = PatientSliceDataset(config["model"]["data"])  # TODO: Afegir al config el path
    train_dataloader = torch.utils.data.DataLoader(train_data, **config["dataloader"])

    model = positional_VQVAE(**config["model"]["vae"])

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    reset_seeds()
    train(model, train_dataloader, image_key, BASE_DIR, num_epochs=config["train"]["num_epochs"], opt=opt)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def VQAutoencoder(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg, structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg["local"])
    cfg["name"] = cfg["model"]["modelname"]
    main(cfg)


if __name__ == "__main__":
    VQAutoencoder()