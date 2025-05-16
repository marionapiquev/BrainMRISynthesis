import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

from VQ-VAE.VQVAEModel_positional import positional_VQVAE, get_time_embedding

from ai4ha.diffusion.models.transformermodel.transformer import DITVideo
from diffusers import DDPMScheduler, UNet2DConditionModel
from dataloader import *
from ai4ha.diffusion.pipelines.pipeline_ddpm_dit_transformer import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

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
from ai4ha.Autoencoders import AutoencoderKL, VQModel

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
import torchvision
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm.auto import trange
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms

from utils_dit import *

from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")



def train(model, train_loader, image_key, BASE_DIR, num_epochs=10, alpha=10, device='cuda', opt=None, num_codes=256,
          visualize_freq=10):
    # Initialize the Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Prepare the model, optimizer, and dataloader with accelerator
    model, opt, train_loader = accelerator.prepare(model, opt, train_loader)

    # Progress bar for tracking iterations
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader),
                            disable=not accelerator.is_local_main_process)
        training_data = {
            "rec_loss": [],
            "cmt_loss": []
        }
        best_loss = 1000
        for i, batch in enumerate(train_loader):

            pixel_values = (batch[image_key].permute(0, 3, 1, 2).to(
                memory_format=torch.contiguous_format).to(accelerator.device))
            opt.zero_grad()

            out, indices, cmt_loss = model(pixel_values)  # Forward pass through the model
            rec_loss = (out - pixel_values).abs().mean()  # Reconstruction loss

            cmt_loss_scalar = cmt_loss.mean()  # Handle multi-element cmt_loss

            # Backpropagation (wrapped by accelerator)
            accelerator.backward(rec_loss + alpha * cmt_loss_scalar)

            opt.step()  # Update model parameters

            progress_bar.set_description(
                f"Epoch {epoch} | " + f"rec loss: {rec_loss.item():.3f} | "
                + f"cmt loss: {cmt_loss_scalar.item():.3f} | "
                + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
            )
            training_data['rec_loss'].append(rec_loss.item())
            training_data['cmt_loss'].append(cmt_loss_scalar.item())

        print(f"Epoch: {epoch} | Reconstruction loss: {np.mean(training_data['rec_loss']):.3f} |  Cmt loss: {np.mean(training_data['cmt_loss']):.3f}")

        if epoch % visualize_freq == 0:
            real_samples, reconstructed_samples = accelerator.gather(pixel_values), accelerator.gather(out)
            visualize_reconstructions(real_samples, reconstructed_samples, epoch, BASE_DIR)

        if np.mean(training_data['rec_loss']) < best_loss:
            best_loss = np.mean(training_data['rec_loss'])
            torch.save(model, f'{BASE_DIR}model/VQVAE-best-model-epoch{epoch}.pt')
            torch.save(model.state_dict(), f'{BASE_DIR}model/VQVAE-best-model-parameters-epoch{epoch}.pt')
            best_model_data = {
                "Epoch": epoch,
                "Reconstruction loss": np.mean(training_data['rec_loss']),
                "Cmt loss": np.mean(training_data['cmt_loss'])
            }

    print("End of Training")
    print(f"BEST MODEL INFO  |  Reconstruction loss: {best_model_data['Reconstruction loss']:.3f}  |  Cmt loss: {best_model_data['Cmt loss']:.3f}  |  Epoch: {best_model_data['Epoch']}")


def train_diffusion(vqvae, scheduler, dit, train_dataloader, optimizer, BASE_DIR, lr_scheduler, config, epochs=30, compute_latents_std=False, visualize_freq=2):
    vqvae.eval()
    accelerator = Accelerator()
    dit, vqvae, optimizer, train_loader, scheduler= accelerator.prepare(dit, vqvae, optimizer, train_dataloader, scheduler)

    if compute_latents_std:
        latents_std(vqvae, train_dataloader, accelerator)

    best_loss = 1000
    for epoch in range(epochs):
        training_data = {
            "loss": []
        }
        for batch in train_loader:
            # Current MRI block
            images = batch["block"].permute(1, 0, 4, 2, 3).to(memory_format=torch.contiguous_format).to(
                accelerator.device)  # (b, s, h, w, c) -> (s,b,h,w,c)
            positions = batch["positions"].permute(1, 0).to(
                memory_format=torch.contiguous_format).to(accelerator.device)

            # Extract latent representations of the current images using the VQVAE
            latents = extract_latents(images, vqvae, positions)

            # Add noise to latents using the diffusion process
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],)).long()
            noised_latents = scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(dit):
                # Pass the noised latents and condition (two previous slices) to the DiT
                pred_noise = dit(noised_latents, timesteps.to(accelerator.device), num_images=0)

                # Calculate loss (MSE between predicted noise and actual noise)
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

                # Backpropagate and optimize
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                training_data['loss'].append(loss.item())

        if accelerator.is_main_process:
            # Print the loss for this epoch
            print(f"Epoch: {epoch} | Loss: {np.mean(training_data['loss']):.5f}")
            if np.mean(training_data['loss']) < best_loss:
                best_loss = np.mean(training_data['loss'])
                torch.save(dit, f'{BASE_DIR}model/Trans-best-model.pt')
                torch.save(dit.state_dict(), f'{BASE_DIR}model/Trans-best-model-parameters.pt')
                best_model_data = {
                    "Epoch": epoch,
                    "Loss": np.mean(training_data['loss'])
                }
                print(f"Saved best model at epoch: {epoch}")
            # Initialize the DDPM Pipline
            ddpm_pipeline = DDPMPipeline(unet=accelerator.unwrap_model(dit), scheduler=scheduler)

            if epoch % visualize_freq == 0:
                evaluate(epoch, ddpm_pipeline, vqvae, BASE_DIR, num_samples=2,
                             accelerator=accelerator)

            # Save best model if needed
            if np.mean(training_data['loss']) < best_loss:
                best_loss = np.mean(training_data['loss'])
                ddpm_pipeline.save_pretrained(f'{BASE_DIR}model/DDPMpipeline-best-model')
                best_model_data = {
                    "Epoch": epoch,
                    "Loss": np.mean(training_data['loss'])
                }




def main(config):
    BASE_DIR = f"{config['base_dir_logs']}{config['name']}/"
    os.makedirs(f"{BASE_DIR}", exist_ok=True)

    DIRS = ["checkpoints", "logs", "samples", "final", "model"]
    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    image_key = ("image" if "image_key" not in config["model"] else
                 config["model"]["image_key"])
    
    # Use a different script to load the data
    print("Starting loading the dataset")
    train_data = PatientSliceDataset(config["model"]["data"]) # TODO: Afegir al config el path
    train_dataloader = torch.utils.data.DataLoader(train_data, **config["dataloader"])
    print("Loaded dataset")
   # Load the VQVAE model 
    vqvae = positional_VQVAE(**config["model"]["vae"])
    if config["pretrainedVQ"]:
        vqvae.load_state_dict(torch.load(config["model"]["pretraiendVQ_path"], weights_only=True))
    else:
        opt = torch.optim.AdamW(vqvae.parameters(), lr=3e-4)
        vqvae = train(vqvae, train_dataloader, image_key, BASE_DIR, num_epochs=config["train"]["num_epochs"], opt=opt)

    # Define the noise scheduler
    scheduler = DDPMScheduler(**config["model"]["scheduler"])

    latent_channels = 8
    image_size = 64
    dit = DITVideo(frame_height=image_size,
                   frame_width=image_size,
                   im_channels=latent_channels,
                   num_frames=7,
                   config=config["model"],
                   **config["model"]["ditv_params"])

    optimizer = torch.optim.Adam(dit.parameters(), lr=1e-4)


    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * 25),
    )

    accparams = config["accelerator"]
    accparams["project_dir"] = BASE_DIR

    if "projectconf" in config:
        accparams["project_config"] = ProjectConfiguration(
            **config["projectconf"])

    print("Start training")
    train_diffusion(vqvae, scheduler, dit, train_dataloader, optimizer, BASE_DIR, lr_scheduler, config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def CondLatentDDPM(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg, structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg["local"])
    cfg["name"] = cfg["model"]["modelname"]
    main(cfg)


if __name__ == "__main__":
    CondLatentDDPM()
    
