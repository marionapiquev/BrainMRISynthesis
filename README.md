# GENERATIVE MODELS FOR BRAIN MRI DATA AUGMENTATION

In this repository, I provide the code used to carry out my final master thesis about "Generative models for brain MRI data augmentation".

## Abstract
Despite the increasing integration of deep learning in medical imaging, the limited availability of annotated datasets remains a critical bottleneck, particularly
for emerging conditions such as Long COVID. Traditional data augmentation
techniques often fail to capture the complexity of 3D brain MRI structures, leading to suboptimal model generalization. In this study, we propose a slice-based
latent diffusion framework designed to synthesize 3D brain MRI volumes in a
computationally efficient slice-by-slice manner. By modeling the joint distribution of MRI slices and incorporating positional embeddings, our approach
maintains spatial coherence across volumes while significantly reducing the
computational burden typically associated with 3D diffusion models. Furthermore, the model is conditioned on diagnostic labels, enabling the generation
of class-specific synthetic data for both healthy individuals and Long COVID
patients. Experiments on classification tasks demonstrate the effectiveness of
the generated synthetic data for augmenting small, imbalanced datasets, ultimately improving the diagnostic performance of deep learning models in the
context of Long COVID classification.

![Generated Samples by our Medical Diffusion model](assets/mri_video.gif)
<p align="center">
  <img src="assets/mri_video.gif" alt="Generated Samples by our Medical Diffusion model" width="400"/>
</p>

## Directories
- `ai4ha`: Contains customized classes adapted from the *Diffusers* library. These are shared components used across all implemented approaches.
- `CondLatentDDPM`: Includes code for training, sampling, and evaluating the Latent Diffusion Model with cross-attention conditioning.
- `LatentDDPM`: Includes code for training, sampling, and evaluating the LDM using additive conditioning.
-  `VQ-VAE`: Contains the implementation for training the VQ-VAE model, which is used as a frozen encoder-decoder component within the LDM pipeline.
- `Transformers`: Provides code for training two different transformer models in an autoregressive fashion for MRI synthesis. *Note: Only training is functional; the sampling implementation is currently buggy.*
