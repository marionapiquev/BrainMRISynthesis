import os
from safetensors.torch import save_model
import shutil
from diffusers import DDPMScheduler, DDIMScheduler
import inspect
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    MultiplicativeLR
from diffusers.optimization import get_scheduler
from safetensors.torch import load_model

SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler
}


def get_diffuser_scheduler(config):
    scheduler = SCHEDULERS[config['diffuser']['type']]

    if 'clip_sample' not in config['diffuser']:
        clip = True
    else:
        clip = config['diffuser']['clip_sample']
    params = {
        'num_train_timesteps': config['diffuser']['num_steps'],
        'beta_schedule': config['diffuser']['beta_schedule'],
        'clip_sample': clip
    }

    if "prediction_type" in set(
            inspect.signature(scheduler.__init__).parameters.keys()):
        params['prediction_type'] = config['diffuser']['prediction_type']
    if ("variance_type" in set(
            inspect.signature(scheduler.__init__).parameters.keys())) and (
                "variance_type" in config['diffuser']):
        params['variance_type'] = config['diffuser']['variance_type']
    if "betas" in config['diffuser']:
        params['beta_start'] = config['diffuser']['betas'][0]
        params['beta_end'] = config['diffuser']['betas'][1]
    if "rescale_betas_zero_snr" in config['diffuser']:
        params['rescale_betas_zero_snr'] = config['diffuser']['rescale_betas_zero_snr']
    return scheduler(**params)


def get_optimizer(model, accelerator, config):
    """_Generate the optimizer object_
    """
    if config['optimizer']['opt'] == 'adam':
        return Adam(model.parameters(),
                    lr=config['optimizer']['learning_rate'] * accelerator.num_processes,
                    betas=(config['optimizer']['beta1'],
                           config['optimizer']['beta2']))
    elif config['optimizer']['opt'] == 'adamw':
        return AdamW(model.parameters(),
                     lr=config['optimizer']['learning_rate'] * accelerator.num_processes,
                     betas=(config['optimizer']['beta1'],
                            config['optimizer']['beta2']),
                     weight_decay=config['optimizer']['weight_decay'],
                     eps=config['optimizer']['epsilon'])


def get_most_recent_checkpoint(logger, BASE_DIR):
    """ _Get the most recent checkpoint_
    Args:
        logger: logger
        BASE_DIR: str
    Returns:
        path: str
    """
    dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    if dirs != []:
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = None
    logger.info(f'CHECKPOINT: {path}')
    return path


def get_best_loss(BASE_DIR, min=True):
    """ _Get the best loss_

    If saved by the training loop, the best loss is saved in a file called 
    `loss.csv` in the `best` directory.
    Args:
        BASE_DIR: str
    Returns:
        path: str
    """
    path = f'{BASE_DIR}/best/loss.csv'
    if os.path.exists(path):
        with open(path, 'r') as f:
            best_loss = float(f.read())
    else:
        if min:
            best_loss = 1e10
        else:
            best_loss = -1e10
    return best_loss


def save_best_model(BASE_DIR, models, loss):
    """_saves the current best model_

    Args:
        logger (_type_): _description_
        BASE_DIR (_type_): _description_
        loss (_type_): _description_
    """
    for name, model in models:
        save_model(model, os.path.join(f'{BASE_DIR}/best/',
                                       f'{name}.safetensors'))
    with open(f'{BASE_DIR}/best/loss.csv', 'w') as f:
        f.write(str(loss))


def load_best_model(BASE_DIR, models):
    """_loads the best model_

    Args:
        logger (_type_): _description_
        BASE_DIR (_type_): _description_
    """
    dmodels = {}
    for name, model in models:
        if os.path.exists(os.path.join(f'{BASE_DIR}/best/', f'{name}.safetensors')):
            load_model(model,
                       os.path.join(f'{BASE_DIR}/best/', f'{name}.safetensors'))
        dmodels[name] = model
    return dmodels


def save_checkpoint_accelerate(logger,
                               BASE_DIR,
                               config,
                               accelerator,
                               global_step,
                               final=False):
    """_save the current model checkpoint_

    """

    if final:
        save_path = os.path.join(f'{BASE_DIR}/model/')
        accelerator.save_state(save_path)
        logger.info(f"Saved final model to {save_path}")
    else:
        save_path = os.path.join(f'{BASE_DIR}/checkpoints/',
                                 f"checkpoint_{global_step:06d}")  # 6 digits
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
        dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
        dirs = sorted([d for d in dirs if d.startswith("checkpoint")])
        if len(dirs) > config["projectconf"]["total_limit"]:
            for d in dirs[:-config["projectconf"]["total_limit"]]:
                logger.info(f'delete {BASE_DIR}/checkpoints/{d}')
                shutil.rmtree(f'{BASE_DIR}/checkpoints/{d}')


def get_lr_scheduler(config, optimizer, dataloader, accparams):
    """_Get the learning rate scheduler_

    Args:
        config (_type_): _description_
        optimizer (_type_): _description_
    Returns:
        _type_: _description_
    """

    if config['lr_scheduler']['type'] == 'plateau':
        return ReduceLROnPlateau(optimizer,
                                 mode=config['lr_scheduler']['mode'],
                                 factor=config['lr_scheduler']['factor'],
                                 patience=config['lr_scheduler']['patience'],
                                 threshold=config['lr_scheduler']['threshold'],
                                 threshold_mode=config['lr_scheduler']['threshold_mode'],
                                 cooldown=config['lr_scheduler']['cooldown'],
                                 min_lr=config['lr_scheduler']['min_lr'],
                                 eps=config['lr_scheduler']['eps'])
    elif config['lr_scheduler']['type'] == 'mutiplicative':
        return MultiplicativeLR(optimizer,
                                lr_lambda=lambda x: config['lr_scheduler']['factor'])
    else:
        return get_scheduler(
            config['lr_scheduler']['type'],
            optimizer=optimizer,
            num_warmup_steps=config["lr_scheduler"]["lr_warmup_steps"] *
            accparams["gradient_accumulation_steps"],
            num_training_steps=config["train"]["num_epochs"]
            #(len(dataloader) * config["train"]["num_epochs"]
            )
