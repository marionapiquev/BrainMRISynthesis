import sys
sys.path.insert(1, "C:/Users/mario/OneDrive/Escritorio/TFM/Codi/mrigen/ai4ha/")
import numpy as np
from GAN.Generators import (
    wavegan_generator_exp_name,
    waveganupsample_generator_exp_name,
    ttsgan_generator_exp_name,
    pulse2pulse_gan_generator_exp_name,
)
from GAN.Discriminators import (
    wavegan_discriminator_exp_name,
    ttsgan_discriminator_exp_name,
)
from Autoencoders import (
    extra_masked_autoencoder_exp_name,
    KLVAE_exp_name,
    VQVAE_exp_name,
)
from Classifier.Transformer import transformer_classifier_exp_name


def experiment_name_diffusion(cfg):
    """
    Returns the experiment name for diffusion models

     DATASET-batch_size-M-diffusion_type-t-num_steps-b-beta-(beta_scheduler)-pt-(prediction_type)-LRS-optimizer
    """

    def channels_growth(cfg):
        g = "1"
        for i in range(1, len(cfg["model"]["params"]["block_out_channels"])):
            g += f"{cfg['model']['params']['block_out_channels'][i]//cfg['model']['params']['block_out_channels'][0]}"
        return g

    def attention_layers(cfg):
        n = 0
        for l in cfg["model"]["params"]["down_block_types"]:
            if "Attn" in l:
                n += 1
        return n

    name = f"{cfg['dataset']['name']}"
    name += f"-b{cfg['dataloader']['batch_size']}"
    name += f"-M-{cfg['model']['modeltype']}"

    if 'UNET' in cfg['model']['modeltype']:
        name += f"-s{cfg['model']['params']['sample_size']}"
        name += (
            f"-l{cfg['model']['params']['block_out_channels'][0]}-{channels_growth(cfg)}"
        )
        name += f"-r{cfg['model']['params']['layers_per_block']}"
        name += f"-a{attention_layers(cfg)}"
    elif 'Transfusion' in cfg['model']['modeltype']:
        name += f"-s{cfg['model']['params']['sample_size']}"
        name += f"-l{cfg['model']['params']['latent_dim']}"
        name += f"-h{cfg['model']['params']['num_heads']}"
        name += f"-r{cfg['model']['params']['num_layers']}"
        name += f"-d{cfg['model']['params']['dropout']}"
        name += f"-f{cfg['model']['params']['ff_size']}"

    if 'act_fn' in cfg['model']['params']:
        name += f"-ac-{cfg['model']['params']['act_fn']}"
    else:
        name += ""  # "-ac-silu"
    if "time_embedding_type" in cfg["model"]["params"]:
        name += f"-te-{cfg['model']['params']['time_embedding_type']}"
    if "class_embed_type" in cfg["model"]["params"]:
        name += f"-ce-{cfg['model']['params']['class_embed_type']}-{cfg['model']['params']['extra_in_channels']}"

    if "time_class_comb" in cfg["model"]["params"]:
        name += f"-tc-{cfg['model']['params']['time_class_comb']}"

    name += f"-DF-{cfg['diffuser']['type']}"
    name += f"-t{cfg['diffuser']['num_steps']}"
    if "betas" in cfg["diffuser"]:
        b1 = np.log10(cfg["diffuser"]["betas"][0])
        b2 = np.log10(cfg["diffuser"]["betas"][1])
        if b1 == int(b1):
            b1 = -int(b1)
        else:
            b1 = str(-int(b1 - 1)) + str(cfg["diffuser"]["betas"][0])[-1]
        if b2 == int(b2):
            b2 = -int(b2)
        else:
            b2 = str(-int(b2 - 1)) + str(cfg["diffuser"]["betas"][1])[-1]
        name += f"-b-{b1}-{b2}"
    if "rescale_betas_zero_snr" in cfg["diffuser"]:
        name += "-ZSNR" if cfg['diffuser']['rescale_betas_zero_snr'] else ""
    name += f"-{cfg['diffuser']['beta_schedule']}"
    name += f"-pt-{cfg['diffuser']['prediction_type']}"
    name += f"-OPT-{cfg['optimizer']['opt']}"
    name += f"-lr{cfg['optimizer']['learning_rate']}"
    name += f"-SC{cfg['lr_scheduler']['type']}"
    if "lr_warmup_steps" in cfg["lr_scheduler"]:
        name += f"-w{cfg['lr_scheduler']['lr_warmup_steps']}"
    name += f"-LS-{cfg['loss']['loss']}"

    return name


def experiment_name_GAN(cfg):
    """
    Returns the experiment name for GANs

    DATASET-batch_size-G-GENERATOR-D-DISCRIMINATOR-OPT-optimizer-lr-learning_rate-LS-loss
    """

    def loss_name(cfg):
        if cfg["loss"]["loss"] == "hinge":
            return "h"
        elif cfg["loss"]["loss"] == "WGAN-GP":
            return f'w{cfg["loss"]["lambda_gp"]}'
        else:
            return "l"

    gen_name = {
        "WaveGAN": wavegan_generator_exp_name,
        "WaveUpGAN": waveganupsample_generator_exp_name,
        "Pulse2Pulse": pulse2pulse_gan_generator_exp_name,
        "TTSGAN": ttsgan_generator_exp_name,
    }

    disc_name = {
        "WaveGAN": wavegan_discriminator_exp_name,
        "TTSGAN": ttsgan_discriminator_exp_name,
    }

    name = f"{cfg['dataset']['name']}"
    name += f"-b{cfg['dataloader']['batch_size']}"
    name += f"-G-{gen_name[cfg['generator']['modeltype']](cfg)}"
    name += f"-D-{disc_name[cfg['discriminator']['modeltype']](cfg)}"
    name += f"-OPT-{cfg['optimizer']['opt']}"
    name += f"-lr{cfg['optimizer']['learning_rate']}"
    name += f"-SC{cfg['lr_scheduler']['type']}"
    if "lr_warmup_steps" in cfg["lr_scheduler"]:
        name += f"-w{cfg['lr_scheduler']['lr_warmup_steps']}"
    name += f"-LS-{loss_name(cfg)}"

    return name


def experiment_name_autoencoder(cfg):
    """
    Returns the experiment name for autoencoders

    DATASET-batch_size-AE-AUTOENCODER-OPT-optimizer-lr-learning_rate-LS-loss
    """

    def loss_name(cfg):
        lname = ''
        if cfg["loss"]["loss"] == "L2":
            lname = "MSE"
        elif cfg["loss"]["loss"] == "L1":
            lname = "MAE"
        else:
            lname = cfg["loss"]["loss"]

        if "perceptual" in cfg["loss"]:
            if not cfg["loss"]["perceptual"]:
                lname += "-NPer"
        return lname

    ae_name = {
        "ExtraMAE": extra_masked_autoencoder_exp_name,
        "KLVAE": KLVAE_exp_name,
        "VQVAE": VQVAE_exp_name,
        "VQGAN": VQVAE_exp_name,
    }

    name = f"{cfg['dataset']['name']}"
    name += f"-b{cfg['dataloader']['batch_size']}"
    name += f"-AE-{ae_name[cfg['model']['modeltype']](cfg)}"
    name += f"-OPT-{cfg['optimizer']['opt']}"
    name += f"-lr{cfg['optimizer']['learning_rate']}"
    name += f"-SC{cfg['lr_scheduler']['type']}"
    if "lr_warmup_steps" in cfg["lr_scheduler"]:
        name += f"-w{cfg['lr_scheduler']['lr_warmup_steps']}"
    name += f"-LS-{loss_name(cfg)}"

    return name


def experiment_name_classifier(cfg):
    """
    Returns the experiment name for classifiers

    DATASET-batch_size-CL-Classifier-OPT-optimizer-lr-learning_rate-LS-loss
    """

    def loss_name(cfg):
        if cfg["loss"]["loss"] == "CE":
            return "CE"
        elif cfg["loss"]["loss"] == "BCE":
            return "BCE"
        elif cfg["loss"]["loss"] == "Ordinal":
            return "OR"
        elif cfg["loss"]["loss"] == "Focal":
            return "FC"
        else:
            return "l"

    ae_name = {"Transformer": transformer_classifier_exp_name}

    name = f"{cfg['dataset']['name']}"
    name += f"-b{cfg['dataloader']['batch_size']}"
    name += f"-CL-{ae_name[cfg['model']['modeltype']](cfg)}"
    name += f"-OPT-{cfg['optimizer']['opt']}"
    name += f"-lr{cfg['optimizer']['learning_rate']}"
    name += f"-SC{cfg['lr_scheduler']['type']}"
    if "lr_warmup_steps" in cfg["lr_scheduler"]:
        name += f"-w{cfg['lr_scheduler']['lr_warmup_steps']}"
    name += f"-LS-{loss_name(cfg)}"

    return name
