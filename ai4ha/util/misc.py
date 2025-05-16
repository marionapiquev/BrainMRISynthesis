from time import time
import importlib
import numpy as np
import torch
import os


def tof(x):
    return 'T' if x else 'F'


def time_management(last_time, qe_time, time_budget, logger):
    """
    Keeps the training time
    """
    epoch_time = (time() - last_time) / 60.0
    last_time = time()
    if len(qe_time) > 10:
        qe_time.pop(0)
    qe_time.append(epoch_time)
    time_budget -= epoch_time
    hours = int(time_budget // 60)
    mins = time_budget - (time_budget // 60) * 60
    logger.info(
        f"** Remaining time budget: {hours:02d}h {mins:3.2f}m - mean iteration time {np.mean(qe_time):3.2f}m **"
    )

    return last_time, qe_time, time_budget


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps.cpu()].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def print_memory(logger, accelerator, where):
    logger.info(
        f"MEM: memory allocated: {torch.cuda.memory_allocated(device=accelerator.device)/(1014*1024)}"
    )
    logger.info(f"MEM: --------- {where} ------------------\n")
    logger.info(f"MEM: \n {torch.cuda.memory_summary(device=accelerator.device)}")
    logger.info("MEM: ---------------------------")


def dataset_first_batch_std(data, config, key):
    config["shuffle"] = False
    config["batch_size"] *= 8
    dataloader = torch.utils.data.DataLoader(data, **config)
    for batch in dataloader:
        images = batch[key]
        break

    return images.std()


def dataset_first_batch_dl_std(dataloader, key):
    for batch in dataloader:
        images = batch[key]
        break

    return images.std()


def dataset_statistics(dataloader, key, channels=3, cuda=True):
    """_Computes mean and std of dataset using a dataloader_

    Args:
        dataloader (_type_): _description_
    """
    dim = list(range(channels))
    dim.remove(1)

    cnt = 0
    if cuda:
        fst_moment = torch.empty(channels).to("cuda")
        snd_moment = torch.empty(channels).to("cuda")
    else:
        fst_moment = torch.empty(channels)
        snd_moment = torch.empty(channels)

    for batch in dataloader:
        images = batch[key]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=dim)
        sum_of_square = torch.sum(images**2, dim=dim)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
    return mean, std


def initialize_dirs(base, dirlist):
    """
    Create directories in a base directory
    """
    for d in dirlist:
        if not os.path.exists(os.path.join(base, d)):
            os.makedirs(os.path.join(base, d), exist_ok=True)


# Functions taken from Latent Diffusion code
def instantiate_from_config(config):
    if "class" not in config:
        raise KeyError("Expected key `class` to instantiate.")
    return get_obj_from_str(config["class"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def log_cond(logger, att, struc):
    if att in struc:
        logger.info(f"{att}: {struc[att]}")  

