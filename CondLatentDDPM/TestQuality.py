import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

from VQ-VAE.VQVAEModel_positional import positional_VQVAE, get_time_embedding
from diffusers import DDPMScheduler, UNet2DModel#, DDPMPipeline
from ai4ha.diffusion.pipelines.cond_pipeline_ddpm_labels import DDPMPipeline
from ai4ha.diffusion.models.unets.cond_unet_2d_labels import UNet2DConditionModel
from dataloader import *
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
import random
import cv2
import os

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

import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
from torchmetrics.image import FrechetInceptionDistance

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")

def normalize_image(image):
    return torch.clamp(((image - image.min()) / (image.max() - image.min())),
                       min=0,
                       max=1)


def extract_latents_vqvae(images, vqvae, pos_emb):
    """
        Extracts latent representations from input images using a VQ-VAE (Vector Quantized Variational Autoencoder) model.
    """
    with torch.no_grad():
        if hasattr(vqvae, 'module'):
            latents = vqvae.module.encode(images, pos_emb)
        else:
            latents = vqvae.encode(images, pos_emb)
        batch_size, channels, height, width = latents.shape
        x = latents.permute(0, 2, 3, 1).contiguous()  # Change to [batch_size, height, width, channels]
        x = x.view(batch_size * height * width, channels)  # Flatten to [batch_size * height * width, channels]
        if hasattr(vqvae, 'module'):
            x, indices, commit_loss = vqvae.module.quantizer(x)
        else:
            x, indices, commit_loss = vqvae.quantizer(x)
        latents = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
    return latents


def evaluate_fid_score(ddpm_pipeline, vqvae, train_dataloader, generated_dataloader, scheduler, accelerator=None):
    """
                Compute the FID score.
    """
    fid = FrechetInceptionDistance(normalize=True, weights_path="/code/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth").to(accelerator.device)
    print("Loading real data")
    for batch in train_dataloader:
        images = normalize_image(batch["image"].permute(0, 3, 1, 2)).to(memory_format=torch.contiguous_format).to(accelerator.device)
        images = ((images + 1) * 0.5).to(torch.uint8)
        if images.shape[1] == 1:  # Check if image is grayscale
            images = images.repeat(1, 3, 1, 1)
        with accelerator.accumulate(fid):
            fid.update(images, real=True)

    print("Loading generated data")
    for batch in generated_dataloader:
        images = normalize_image(batch["image"]).to(memory_format=torch.contiguous_format).to(accelerator.device)
        images = ((images + 1) * 0.5).to(torch.uint8)
        if images.shape[1] == 1:  # Check if image is grayscale
            images = images.repeat(1, 3, 1, 1)
        with accelerator.accumulate(fid):
            fid.update(images, real=False)

    if accelerator.is_main_process:
        print(f"Computed FID: {fid.compute()}")


def evaluate_supervised_metrics(ddpm_pipeline, vqvae, test_dataloader, scheduler, accelerator, BASE_DIR):
    """
                    Compute the SSIM and PSNR scores.
    """
    device = accelerator.device if accelerator else "cuda"

    ssim_metric = StructuralSimilarityIndexMeasure().to(accelerator.device)
    psnr_metric = PeakSignalNoiseRatio().to(accelerator.device)

    ssim_scores = []
    psnr_scores = []

    first_batch = True
    image_path = f"{BASE_DIR}samples/comparison.png"

    for batch in test_dataloader:
        images = batch["image"].permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).to(accelerator.device)

        positions = batch["position"].to(memory_format=torch.contiguous_format).to(accelerator.device)

        label = batch["label"].to(memory_format=torch.contiguous_format).to(accelerator.device)

        pos_emb = get_time_embedding(positions, temb_dim=190)
        latents = extract_latents_vqvae(images, vqvae, pos_emb)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],)).long()
        noised_latents = scheduler.add_noise(latents, noise, timesteps)

        generated_images = ddpm_pipeline(batch_size=8,
                                        output_type='np.array', num_inference_steps=100, latent=True, vae=vqvae,
                                        position=pos_emb, class_cond=(positions, label), mode="evaluate", accelerator= accelerator, input_image=noised_latents).images

        # Ensure generated images are on the same device
        generated_images = generated_images.to(accelerator.device)

        print("Decoded images range: min =", generated_images.min().item(), ", max =", generated_images.max().item(),
              "Shape:", generated_images.shape)
        print("Target images range:  min =", images.min().item(), ", max =", images.max().item(), "Shape:",
              images.shape)
        print("Decoded images range: min =", generated_images.min().item(), ", max =", generated_images.max().item(),
              "Shape:", generated_images.shape)
        print("Target images range:  min =", images.min().item(), ", max =", images.max().item(), "Shape:",
              images.shape)
        
        # Compute SSIM and PSNR
        ssim_value = ssim_metric(generated_images, images)
        psnr_value = psnr_metric(generated_images, images)

        ssim_scores.append(ssim_value.item())
        psnr_scores.append(psnr_value.item())

        if first_batch:
            first_batch = False  # Prevent further plotting

            # Convert to CPU for plotting
            images_np = images.cpu().permute(0, 2, 3, 1).numpy()
            generated_images_np = generated_images.cpu().permute(0, 2, 3, 1).numpy()

            fig, axes = plt.subplots(nrows=2, ncols=len(images_np), figsize=(15, 6))

            for i in range(len(images_np)):
                # Original Image
                axes[0, i].imshow(images_np[i])
                axes[0, i].axis("off")
                axes[0, i].set_title(f"Original {i + 1}")

                # Generated Image
                axes[1, i].imshow(generated_images_np[i])
                axes[1, i].axis("off")
                axes[1, i].set_title(f"Generated {i + 1}")

            plt.suptitle("Original vs. Generated Images", fontsize=16)
            plt.axis("off")
            plt.savefig(image_path)
            plt.close()

    if accelerator.is_main_process:
        # Compute final averages
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_psnr = sum(psnr_scores) / len(psnr_scores)

        print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f} dB")


def main(config, num_mris=2, evaluation_type = "unsupervised"):
    BASE_DIR = f"{config['base_dir_logs']}{config['name']}/"
    os.makedirs(f"{BASE_DIR}", exist_ok=True)

    DIRS = ["samples"]
    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    image_key = ("image" if "image_key" not in config["evaluate"] else
                 config["evaluate"]["image_key"])

    # Load the test set data
    test_data = PatientSliceDataset(config["model"]["testdata"])
    test_dataloader = torch.utils.data.DataLoader(test_data, **config["dataloader"])

    #Load the LDM pipeline
    ddpm_pipeline = DDPMPipeline.from_pretrained(config["evaluate"]["path_pipeline"], use_safetensors=True)

    vqvae = positional_VQVAE(**config["evaluate"]["vae"])
    vqvae.load_state_dict(torch.load(config["evaluate"]["pretraiendVQ_path"], weights_only=True))
    vqvae.eval()

    scheduler = DDPMScheduler(
        **config["evaluate"]["scheduler"])

    accelerator = Accelerator()
    scheduler = ddpm_pipeline.scheduler
    (vqvae, ddpm_pipeline, test_dataloader, scheduler) = accelerator.prepare(vqvae, ddpm_pipeline, test_dataloader, scheduler)

    if evaluation_type == "unsupervised":
        model_id = config["evaluate"]["model_id"]
        # Load the train set data
        train_data = PatientSliceDataset(config["model"]["traindata"])
        train_dataloader = torch.utils.data.DataLoader(train_data, **config["dataloader"])

        # Load the generated data
        generated_data = PatientSliceDatasetGenerated(
            f"{config["model"]["generateddata"]}/{model_id}")
        # Use the same amount of train and generated data
        random_indices = random.sample(range(len(generated_data)), 33424)
        generated_data = Subset(generated_data, random_indices)
        generated_dataloader = torch.utils.data.DataLoader(generated_data, **config["dataloader"])
        
        evaluate_fid_score(ddpm_pipeline, vqvae, train_dataloader, generated_dataloader, scheduler, accelerator=accelerator)
    else:

        evaluate_supervised_metrics(ddpm_pipeline, vqvae, test_dataloader, scheduler, accelerator=accelerator, BASE_DIR=BASE_DIR)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def sample(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg, structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg["local"])
    cfg["name"] = cfg["evaluate"]["modelname"]
    main(cfg)


if __name__ == "__main__":
    sample()
    
