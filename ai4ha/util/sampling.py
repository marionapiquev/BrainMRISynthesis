import numpy as np
import torch
from tqdm import tqdm


def sampling_diffusion(config,
                       BASE_DIR,
                       pipeline,
                       generator,
                       conditioned=False,
                       nsamp=1,
                       epoch=0,
                       final=False,
                       generate=False):
    """_Sample from the diffusion model_
    """

    def gen_labels(final, generate):
        if final:
            return torch.randint(
                0, config['dataset']['nclasses'],
                (config['dataloader']['batch_size'], )).to(
                    dtype=torch.long).to(device=pipeline.device)
        elif generate:
            return torch.randint(
                0, config['dataset']['nclasses'],
                (config['samples']['sample_batch_size'], )).to(
                    dtype=torch.long).to(device=pipeline.device)
        else:
            return torch.Tensor(range(config['dataset']['nclasses'])).to(
                dtype=torch.long).to(device=pipeline.device)

    samples, samples_labs = [], []

    for g in range(nsamp):
        labels = None if not conditioned else gen_labels(final, generate)
        batch_size = 1 if not conditioned else labels.shape[0]
        # run pipeline in inference (sample random noise and denoise)
        data = pipeline(
            generator=generator,
            class_cond=labels,
            batch_size=batch_size,
            num_inference_steps=config['diffuser']['num_inference_steps'],
            output_type="numpy").images

        samples.append(data.cpu().numpy())
        if conditioned:
            samples_labs.append(labels.cpu().numpy())
        else:
            samples_labs.append([0])
    suffix = '' if final else f'_{epoch:05d}'
    if generate:
        suffix = f'_{config["name"]}_{len(samples)}'
    np.savez_compressed(f"{BASE_DIR}/samples/sampled_data{suffix}.npz",
                        samples=np.concatenate(samples),
                        classes=np.concatenate(samples_labs))


def sampling_GAN(config,
                 BASE_DIR,
                 accelerator,
                 netG,
                 latent_dim,
                 nsamp,
                 epoch,
                 conditioned=False,
                 final=False):
    """_sampling from the GAN generator_
    """
    llabels = []
    lsamples = []
    for i in tqdm(range(nsamp)):
        noise = torch.randn(*tuple([1] + latent_dim)).to(accelerator.device)
        label = torch.tensor([i % config['generator']['params']['n_classes']
                              ]).long().to(accelerator.device)
        if conditioned:
            label = torch.tensor([
                i % config['generator']['params']['n_classes']
            ]).long().to(accelerator.device)
            fake = netG(noise, label)
            llabels.append(label.cpu().detach().numpy())
        else:
            fake = netG(noise)
        lsamples.append(fake.cpu().detach().numpy())

    suffix = '' if final else f'_{epoch:05d}'
    if conditioned:
        np.savez(f"{BASE_DIR}/samples/sampled_data{suffix}.npz",
                 samples=np.concatenate(lsamples),
                 classes=np.concatenate(llabels))
    else:
        np.savez(f"{BASE_DIR}/samples/sampled_data{suffix}.npz",
                 samples=np.concatenate(lsamples))


def sampling_MAE(
    BASE_DIR,
    accelerator,
    model,
    dataset,
    nsamp,
):
    """ _Samplig from the ExtraMAE model_

    Passes random samples through the model, reconstructs the data and saves
    the samples and labels.
    """
    lsamples = []
    llabels = []
    lorig = []
    lmask = []
    # lpred = []
    samples = np.random.randint(0, len(dataset), (nsamp, ))
    for i in tqdm(samples):
        data, label = dataset[i]
        llabels.append(label)
        lorig.append(np.expand_dims(data, 0))
        pred, mask = model(
            torch.tensor(data).unsqueeze(0).float().to(accelerator.device))
        pred = pred.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        data = data * np.abs(mask - 1)
        pred_mask = pred * mask
        pred_mask = pred_mask + data
        lsamples.append(pred_mask)
        lmask.append(mask)

    np.savez(f"{BASE_DIR}/samples/sampled_data.npz",
             samples=np.concatenate(lsamples),
             orig=np.concatenate(lorig),
             classes=llabels,
             mask=np.concatenate(lmask))
