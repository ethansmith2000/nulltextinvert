# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, List, Optional, Union

import torch

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from packaging import version

import inspect
from typing import Callable, List, Optional, Union

import torch

from diffusers.utils import is_accelerate_available
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer,

from PIL import Image
import numpy as np
import torchvision.transforms as T

import re

import math

from diffusers import StableDiffusionPipeline

import copy

import glob

from p2p import ptp_utils, nulltextinvert
from p2p.prompt2prompt import make_controller

def preprocess(image, w, h):
    image = image.convert(mode="RGB")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class MyPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
            requires_safety_checker: bool = False,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker,
                         feature_extractor)  # TODO will this override anything?

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.scheduler_map = {
            "klms": LMSDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler
        }

        self.inv_memory = {
            "inv_latent": None,
            "inv_embeddings": None,
            "init": None,
            "gt_prompt": None,
            "steps": None,
            "guidance_scale": None,
            "num_inner_steps": None,
        }


    def set_new_scheduler(self, name):
        self.scheduler = self.scheduler_map[name](beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        # TODO these params may not work for every scheduler

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return image

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
                        dupe_seed=False):
        if dupe_seed:
            shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        else:
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        if dupe_seed:
            latents = latents.repeat(batch_size, 1, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_init(self, init_image, batch_size, height, width, timestep, dtype, device, generator, dupe_seed=False):
        if isinstance(init_image, torch.Tensor):  # if init is already latent object, just resize
            latents = init_image.to(device=self.device, dtype=dtype)
            resize = T.Resize((height // 8, width // 8))
            latents = resize(latents)
            noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
            # get latents
            latents = self.scheduler.add_noise(latents, noise, timestep)
            # latents = torch.cat([latents] * batch_size, dim=0)
        else:  # or path
            self.vae = self.vae.to(dtype)
            init = preprocess(init_image, height, width).to(device=self.device, dtype=dtype)
            init_latent_dist = self.vae.encode(init).latent_dist
            latents = init_latent_dist.sample(generator=generator)
            latents = 0.18215 * latents
            latents = torch.cat([latents] * batch_size, dim=0)
            # add noise to latents using the timesteps
            if dupe_seed:
                noise = torch.randn((1, latents.shape[1], latents.shape[2], latents.shape[3]), generator=generator,
                                    device=device, dtype=dtype)
                noise = noise.repeat(batch_size, 1, 1, 1)
            else:
                noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
            # get latents
            latents = self.scheduler.add_noise(latents, noise, timestep)

        return latents

    def run_null_inversion(self, init, width, height, gt_prompt, steps, guidance_scale, num_inner_steps=None,
                           early_stop_epsilon=1 - 5):
        image = preprocess(init, width, height).to(self.unet.device).to(self.unet.dtype)
        latent_gt = self.vae.encode(image).latent_dist.sample().float()  # TODO we might need to use mean instead
        latent_gt = 0.18215 * latent_gt
        if not latent_gt.ndim == 4:
            latent_gt = latent_gt.unsqueeze(0)

        orig_dtype = self.unet.dtype
        self.unet = self.unet.to(torch.float32)
        self.text_encoder = self.text_encoder.to(torch.float32)
        self.vae = self.vae.to(torch.float32)

        inv_latent, inv_embeddings = nulltextinvert.invert(self, latent_gt, gt_prompt, steps, guidance_scale,
                                                           verbose=False, num_inner_steps=num_inner_steps,
                                                           early_stop_epsilon=early_stop_epsilon, orig_dtype=orig_dtype)
        self.unet = self.unet.to(orig_dtype)
        self.text_encoder = self.text_encoder.to(orig_dtype)
        self.vae = self.vae.to(orig_dtype)
        self.inv_memory = {
            "init": init,
            "gt_prompt": gt_prompt,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "num_inner_steps": num_inner_steps,
            "inv_latent": inv_latent.to(orig_dtype),
            "inv_embeddings": inv_embeddings,
        }

    def prompt2prompt(
            self,
            prompt: List[str],
            controller_params,
            num_inference_steps: int = 50,
            guidance_scale: Optional[float] = 7.5,
            width=512,
            height=512,
            generator: Optional[torch.Generator] = None,
            latent: Optional[torch.FloatTensor] = None,
            uncond_embeddings=None,
    ):
        strength = 1.0
        batch_size = len(prompt)  # bs > 1
        controller = make_controller(*controller_params[0], blend_words=controller_params[1],
                                     equalizer_params=controller_params[2], device=self.unet.device,
                                     dtype=self.unet.dtype)
        ptp_utils.register_attention_control(self, controller)

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].to(self.unet.dtype)
        max_length = text_input.input_ids.shape[-1]
        # if uncond_embeddings is None:
        #     uncond_input = self.tokenizer(
        #         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        #     )
        #     uncond_embeddings_backup = self.text_encoder(uncond_input.input_ids.to(self.unet.device))[0].to(
        #         self.unet.dtype)
        # else:
        #     uncond_embeddings_backup = None

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.unet.device)
        timesteps = self.get_timesteps(num_inference_steps, strength, self.unet.device)

        latent, latents = ptp_utils.init_latent(latent, self, height, width, generator, batch_size)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # if uncond_embeddings_backup is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            # else:
            #     context = torch.cat([uncond_embeddings_backup, text_embeddings])
            latents = ptp_utils.diffusion_step(self, controller, latents, context, t, guidance_scale,
                                               low_resource=False)

        return latents

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            init=None,
            strength: Optional[float] = 0.5,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            scheduler: Optional[str] = None,



            prompt2prompt_mode=False,
            gt_prompt=None,
            num_inner_steps=10,
            cross_replace_steps={'default_': .8, },
            self_replace_steps=.6,
            blend_word=None,  # ((('cat',), ("cat",))),  # for local edit
            eq_params=None,  # {"words": ("silver", 'sculpture',), "values": (2, 2,)},
            is_replace_controller=False,

    ):

        self.scheduler.set_timesteps(num_inference_steps, device=self.unet.device)
        assert batch_size == 1 and len(prompt) == 1 and init is not None, "you done messed up"
        assert isinstance(self.scheduler, DDIMScheduler)
        if self.inv_memory["inv_latent"] is not None or self.inv_memory["inv_embeddings"] is not None:
            if (self.inv_memory["init"] == init and
                    self.inv_memory["gt_prompt"] == gt_prompt and
                    self.inv_memory["steps"] == num_inference_steps and
                    self.inv_memory["guidance_scale"] == guidance_scale and
                    self.inv_memory["num_inner_steps"] == num_inner_steps):
                new_run = False
            else:
                new_run = True

        else:
            new_run = True

        if new_run:
            self.run_null_inversion(init, width, height, gt_prompt, num_inference_steps, guidance_scale,
                                    num_inner_steps=num_inner_steps)

        p2pprompts = [gt_prompt, prompt[0]]
        controller_params = (
            (p2pprompts, self.tokenizer, num_inference_steps, is_replace_controller, cross_replace_steps,
             self_replace_steps),
            blend_word,
            eq_params)
        latent_output = self.prompt2prompt(p2pprompts, controller_params, width=width, height=height, num_inference_steps=num_inference_steps,
                                           guidance_scale=guidance_scale, latent=self.inv_memory["inv_latent"],
                                           uncond_embeddings=self.inv_memory["inv_embeddings"])

        self.vae = self.vae.to(self.unet.dtype)
        image = self.decode_latents(latent_output)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)