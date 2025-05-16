import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

from VQ-VAE.VQVAEModel_positional import positional_VQVAE, get_time_embedding
from diffusers import DDPMScheduler, UNet2DModel#, DDPMPipeline
from ai4ha.diffusion.pipelines.cond_pipeline_ddpm_labels import DDPMPipeline
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
import time

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")


def generate_full_mri(ddpm_pipeline, vqvae, BASE_DIR, model_id, label=0, sample_size=1, accelerator=None, fps=5):
    """
        Evaluates the DDPM pipeline's generation performance by generating and saving a full MRI volume.
    """
    device = accelerator.device if accelerator else "cuda"
    decoded_images = []
    decoded_images_to_image = []

    while True:
        seed = random.randint(0, 1000)
        npz_path = os.path.join("/gpfs/projects/bsc70/bsc152213/generated_data/", f"{model_id}/{label}_{seed}_struc_brain.npz")
        if not os.path.exists(npz_path):
            break  # Found a unique seed, exit the loop

    positions = list(range(0, 181))
    image_path = f"{BASE_DIR}samples/{label}_video_sample{seed}.png"
    video_path = os.path.join(BASE_DIR, f"samples/video_sample{seed}.mp4")
    sampling_time = []
    for i in range(len(positions)):
        start_time = time.time()
        pos_emb = get_time_embedding(torch.tensor([positions[i]]), temb_dim=190)
        generated_image = ddpm_pipeline(batch_size=1, generator=torch.Generator(device='cpu').manual_seed(seed),
                                         output_type='np.array', num_inference_steps=100, latent=True, vae=vqvae.to(device),
                                         position=pos_emb.to(device), class_cond=(torch.tensor([positions[i]]), torch.tensor([label])), accelerator=accelerator).images
        image_array = np.array(generated_image[0].detach().cpu().numpy())  # Ensure it's a NumPy array
        decoded_images.append(image_array)
        decoded_images_to_image.append(generated_image[0])
        end_time = time.time()
        sampling_time.append(end_time - start_time)


    decoded_images = np.stack(decoded_images, axis=0)  # (sample_size, H, W)
    decoded_images_2 = torch.stack(decoded_images_to_image, axis=0)

    print(f"MRI shape: {decoded_images.shape}")
    print(f"Average slices sampling time: {np.mean(sampling_time):.2f} seconds")
    print(f"Total mri sampling time: {np.sum(sampling_time):.2f} seconds")

    np.savez_compressed(npz_path, images=decoded_images)

    # Create a grid of images and save
    grid = torchvision.utils.make_grid(decoded_images_2, nrow=14, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Samples")
    plt.axis("off")
    plt.savefig(image_path)
    plt.close()


def generate_samples(ddpm_pipeline, vqvae, BASE_DIR, label=0, sample_size=16, accelerator=None):
    """
        Evaluates the DDPM pipeline's generation performance by generating and saving a grid of samples every 10 slides.
    """
    device = accelerator.device if accelerator else "cuda"
    decoded_images = []

    seed = random.randint(0, 1000)
    positions = [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153]
    if label == 1:
        image_path = f"{BASE_DIR}samples/02_sample{seed}.png"
    else:
        image_path = f"{BASE_DIR}samples/01_sample{seed}.png"
    for i in range(sample_size):
        pos_emb = get_time_embedding(torch.tensor([positions[i]]), temb_dim=190)
        generated_image = ddpm_pipeline(batch_size=1, generator=torch.Generator(device='cpu').manual_seed(seed),
                                         output_type='np.array', num_inference_steps=100, latent=True, vae=vqvae.to(device),
                                         position=pos_emb.to(device), class_cond=(torch.tensor([positions[i]]), torch.tensor([label])), accelerator=accelerator).images
        decoded_images.append(generated_image[0])

    # Stack all generated images into a tensor
    decoded_images = torch.stack(decoded_images, axis=0)

    print('Samples')
    print(torch.max(decoded_images))
    print(torch.min(decoded_images))
    print(decoded_images.shape)

    # Create a grid of images and save
    grid = torchvision.utils.make_grid(decoded_images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Samples")
    plt.axis("off")
    plt.savefig(image_path)
    plt.close()

def main(config, num_mris=50, type="full_mri"):
    BASE_DIR = f"{config['base_dir_logs']}{config['name']}/"
    os.makedirs(f"{BASE_DIR}", exist_ok=True)

    DIRS = ["samples"]
    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)
    print(config["sample"]["path_pipeline"])
    ddpm_pipeline = DDPMPipeline.from_pretrained(config["sample"]["path_pipeline"], use_safetensors=True)

    vqvae = positional_VQVAE(**config["sample"]["vae"])
    vqvae.load_state_dict(torch.load(config["sample"]["pretraiendVQ_path"], weights_only=True))
    vqvae.eval()
    accelerator = Accelerator()
    (vqvae, ddpm_pipeline) = accelerator.prepare(vqvae, ddpm_pipeline)
    labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    type = config["sample"]["type"]
    for mri in range(num_mris):
        if type == "full_mri":
            print(config["sample"]["model_id"])
            generate_full_mri(ddpm_pipeline, vqvae, BASE_DIR, model_id=config["sample"]["model_id"], label=labels[mri], sample_size=1, accelerator=accelerator, fps=5)
        else:
            generate_samples(ddpm_pipeline, vqvae, BASE_DIR, label=labels[mri], sample_size=16,
                         accelerator=accelerator)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def sample(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg, structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg["local"])
    cfg["name"] = cfg["sample"]["modelname"]
    main(cfg)


if __name__ == "__main__":
    sample()
    
