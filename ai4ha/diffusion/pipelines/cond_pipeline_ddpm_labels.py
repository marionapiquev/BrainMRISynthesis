# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @03/23/2023
# Fixing DDPM pipeline so it receives also a class for class conditioning generation

from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self,
                 batch_size: int = 1,
                 class_cond: int = None,
                 generator: Optional[Union[torch.Generator,
                                           List[torch.Generator]]] = None,
                 num_inference_steps: int = 1000,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 latent=False,
                 vae=None,
                 accelerator=None,
                 position=None,
                 mode="sampling",
                 input_image=None) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        if mode == "evaluate":
            image = input_image.to(accelerator.device) #.to(self.device)
        else:
            # Sample gaussian noise to begin loop
            if isinstance(self.unet.sample_size, int):
                image_shape = (
                    batch_size,
                    self.unet.config.out_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (batch_size, self.unet.config.out_channels,
                               *self.unet.config.sample_size)

            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(accelerator.device) #.to(self.device)
            else:
                image = randn_tensor(image_shape,
                                     generator=generator,
                                     device=accelerator.device)
                                     #device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            label = None if class_cond is None else class_cond
            if label is not None:
                label1, label2 = label[0].to(accelerator.device), label[1].to(accelerator.device)
            t = t.to(accelerator.device)
            self.unet = self.unet.to(accelerator.device)
            model_output = self.unet(image, t,  class_labels=label1, encoder_hidden_states=None, added_cond_kwargs={'labels':label2.unsqueeze(1)}).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output,
                                        t,
                                        image,
                                        generator=generator).prev_sample

        if not latent:
            image = (image / 2 + 0.5).clamp(0, 1)
        else:
            if vae is not None:
                if hasattr(vae, 'module'):
                    image = vae.module.decode(image.to(accelerator.device), position.to(accelerator.device))
                else:
                    image = vae.decode(image.to(accelerator.device), position.to(accelerator.device))
                image = image.clamp(-1, 1)
            else:
                pass
                # print('$$$$$$$$$$$$$$$$$$$$$$$ no decoding sampling')

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, )

        return ImagePipelineOutput(images=image)
