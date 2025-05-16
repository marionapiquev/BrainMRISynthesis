import sys
sys.path.insert(1, "/home/bsc/bsc152213/mrigen/ai4ha/")
from omegaconf import OmegaConf
import json
from ..paths_config import LOCAL_REMOTE_PATHS_CONFIG
import yaml


def load_config(nfile):
    """
    Loads configuration file
    """

    ext = ".json" if "json" not in nfile else ""

    f = open(nfile + ext, "r")
    return json.load(f)


def load_config_yaml(nfile):
    """
    Loads configuration file
    """

    ext = ".yaml" if "yaml" not in nfile else ""

    f = open(nfile + ext, "r")
    return yaml.load(f, Loader=yaml.Loader)


def load_config_fix_paths(nfile, local=False):
    """
    Loads configuration file
    """

    ext = ".json" if "json" not in nfile else ""

    f = open(nfile + ext, "r")
    config = fix_paths(json.load(f), local)
    return config


def load_jobs(jobsfile):
    """ loads a file with job configurations"""
    ext = ".jobs" if "jobs" not in jobsfile else ""
    f = open(jobsfile + ext, "r")
    ljobs = []
    for line in f:
        if line[0] not in ["#", '@']:
            ljobs.append(line.replace('\n', ''))
    return ljobs


def save_config(config, nfile):
    """
    Saves configuration file
    """
    f = open(nfile, "w")
    OmegaConf.save(config, f=f)
    f.close()


def fix_paths(config, local=False):

    def fix_dataset_paths(dataset, config):
        for key in config[dataset].keys():
            if ('train' in key) or ('test' in key) or ('val' in key):
                if 'filename' in config[dataset][key]['params']:
                    path_par = 'filename'
                elif 'dir' in config[dataset][key]['params']:
                    path_par = 'dir'
                elif 'data_root' in config[dataset][key]['params']:
                    path_par = 'data_root'
                else:
                    path_par = None

                if path_par is not None:
                    config[dataset][key]['params'][
                        path_par] = LOCAL_REMOTE_PATHS_CONFIG['Data'][
                            path] + config[dataset][key]['params'][path_par]

    # Fix experiment path
    path = 'local_path' if local else 'remote_path'
    config['exp_dir'] = LOCAL_REMOTE_PATHS_CONFIG['Experiment'][path] + config[
        'exp_dir']
    # Fix dataset paths
    if 'dataset' in config:
        fix_dataset_paths('dataset', config)
    if 'generated' in config:
        fix_dataset_paths('generated', config)

    # Fix path in losses with pretrained models
    if not local:
        if 'loss' in config:
            if 'params' in config['loss']:
                if 'params' in config['loss']['params']:
                    if 'root' in config['loss']['params']['params']:
                        config['loss']['params']['params'][
                            'root'] = LOCAL_REMOTE_PATHS_CONFIG['Loss']['root']

    # Fix path for training that need pretrained models
    if 'latents' in config:
        if 'model' in config['latents']:
            config['latents']['model'] = LOCAL_REMOTE_PATHS_CONFIG['Models'][
                path] + config['latents']['model']
        if 'norm_path' in config['latents']:
            config['latents']['norm_path'] = LOCAL_REMOTE_PATHS_CONFIG['Data'][
                path] + config['latents']['norm_path']
    return config


def adapt_hydra_optuna_sweep(config):
    """_summary_

     Takes the parameters of the hydra optuna sampling and copies them to 
     the corresponding parameters of the training configuration

     Parameters have a name in the form "sectionname-sectionname-parametername" to 
     match the keys of the configuration
    """
    for key in config:
        if '-' in key:
            params = key.split("-")
            if len(params) == 2:
                config[params[0]][params[1]] = config[key]
            elif len(params) == 3:
                config[params[0]][params[1]][params[2]] = config[key]
            else:
                print("Error: key not recognized")
    return config
