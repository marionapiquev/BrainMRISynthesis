import torch
from tqdm.auto import trange
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision
import torch.nn as nn
import random
from VQ-VAE.VQVAEModel_positional import positional_VAE, get_time_embedding


def extract_latents(images, vqvae, positions):
    """
        Extracts latent representations from input images using a VQ-VAE (Vector Quantized Variational Autoencoder) model.
    """
    block_size, _, _, _, _ = images.shape
    latent_blocks = []
    for i in range(block_size):
        pos_emb = get_time_embedding(positions[i], temb_dim=190)
        with torch.no_grad():
            if hasattr(vqvae, 'module'):
                latents = vqvae.module.encode(images[i], pos_emb)
            else:
                latents = vqvae.encode(images[i], pos_emb)
            batch_size, channels, height, width = latents.shape
            x = latents.permute(0, 2, 3, 1).contiguous()  # Change to [batch_size, height, width, channels]
            x = x.view(batch_size * height * width, channels)  # Flatten to [batch_size * height * width, channels]
            if hasattr(vqvae, 'module'):
                x, indices, commit_loss = vqvae.module.quantizer(x)
            else:
                x, indices, commit_loss = vqvae.quantizer(x)
            latents = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
            latent_blocks.append(latents)
    latent_block = torch.stack(latent_blocks, axis=0)
    return torch.tensor(latent_block).permute(1,0,2,3,4)  # (batch, num_channels, num_frames, height, width)



def visualize_reconstructions(real_samples, reconstructed_samples, iteration, BASE_DIR, num_samples=4):
    """
        Visualizes and saves a comparison between real and reconstructed samples, to monitor
        the reconstruction performance of the vqvae.
    """
    real_samples = real_samples[:num_samples].detach().cpu()
    reconstructed_samples = reconstructed_samples[:num_samples].detach().cpu()
    path = f"{BASE_DIR}samples/E{iteration:04d}-S{iteration + 1:05d}s.png"
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))

    for i in range(num_samples):
        axes[0, i].imshow(real_samples[i].permute(1, 2, 0),
                          cmap=plt.get_cmap('gray'))
        axes[0, i].axis('off')
        axes[0, i].set_title("Real")

        axes[1, i].imshow(reconstructed_samples[i].permute(1, 2, 0), cmap=plt.get_cmap('gray'))
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    fig.savefig(path)


def latents_std(vqvae, train_dataloader, accelerator, preloop=50):
    """
        Calculates and adjusts the scaling factor of a VQ-VAE model based on the standard deviation of its latent
        representations over a number of training samples.
    """
    vqvae.scaling_factor = 1.0
    # Preloop measuring of stdev of the latent
    latent_std = 0
    for step, batch in enumerate(train_dataloader):
        clean_images = batch['image'].permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).to(
            accelerator.device)
        if hasattr(vqvae, 'module'):
            clean_images_latents = vqvae.module.encoder(clean_images)
        else:
            clean_images_latents = vqvae.encoder(clean_images)
        latent_std += clean_images_latents.detach().flatten().std()
        if step > preloop:
            break

    vqvae.scaling_factor = 1.0 / (latent_std / preloop)
    print(1.0 / (latent_std / preloop))


def evaluate(iteration, ddpm_pipeline, vqvae, BASE_DIR, num_samples=2, accelerator=None):
    """
        Evaluates the DDPM pipeline's generation performance.
    """
    device = accelerator.device if accelerator else "cuda"
    for i in range(num_samples):
        seed = random.randint(0, 100)
        image_path = f"{BASE_DIR}samples/sample{seed}.png"
        # Generate new images using the combined condition
        generated_image = ddpm_pipeline(
            batch_size=1,
            in_channels=8,
            sample_size=64,
            generator=torch.Generator(device='cpu').manual_seed(10),
            output_type='np.array',
            num_inference_steps=1000,
            latent=True,
            vae=vqvae
        ).images

        tv_frames = generated_image * 255

        tv_frames = tv_frames.permute((0, 2, 3, 1))
        if tv_frames.shape[-1] == 1:
            tv_frames = tv_frames.repeat((1, 1, 1, 3))

        torchvision.io.write_video(image_path,
                                   tv_frames,
                                   fps=5)
