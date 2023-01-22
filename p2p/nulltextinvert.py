from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as F
import numpy as np
import abc
import p2p.ptp_utils as ptp_utils
import p2p.seq_aligner as seq_aligner
from torch.optim.adam import Adam
from PIL import Image


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def inversion_prev_step(whole_model, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    prev_timestep = timestep - whole_model.scheduler.config.num_train_timesteps // whole_model.scheduler.num_inference_steps
    alpha_prod_t = whole_model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = whole_model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else whole_model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample


def inversion_next_step(whole_model, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(
        timestep - whole_model.scheduler.config.num_train_timesteps // whole_model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = whole_model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else whole_model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = whole_model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(whole_model, latents, t, context):
    noise_pred = whole_model.unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


def get_noise_pred(whole_model, text_embeds, latents, t, guidance_scale, is_forward=True, context=None):
    latents_input = torch.cat([latents] * 2)
    if context is None:
        context = text_embeds
    guidance_scale = 1 if is_forward else guidance_scale
    noise_pred = whole_model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    if is_forward:
        latents = inversion_next_step(whole_model, noise_pred, t, latents)
    else:
        latents = inversion_prev_step(whole_model, noise_pred, t, latents)
    return latents


@torch.no_grad()
def latent2image(whole_model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = whole_model.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def image2latent(whole_model, image, device, dtype):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device).to(dtype)
            latents = whole_model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


@torch.no_grad()
def init_prompt(whole_model, prompt: str, device, dtype):
    uncond_input = whole_model.tokenizer(
        [""], padding="max_length", max_length=whole_model.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = whole_model.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype)
    text_input = whole_model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=whole_model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = whole_model.text_encoder(text_input.input_ids.to(device))[0].to(dtype)
    text_embeds = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeds


@torch.no_grad()
def ddim_loop(whole_model, text_embeds, latent, num_ddim_steps):
    uncond_embeddings, cond_embeddings = text_embeds.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in range(num_ddim_steps):
        t = whole_model.scheduler.timesteps[len(whole_model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(whole_model, latent, t, cond_embeddings)
        latent = inversion_next_step(whole_model, noise_pred, t, latent)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(whole_model, text_embeds, latent, num_ddim_steps):
    ddim_latents = ddim_loop(whole_model, text_embeds, latent, num_ddim_steps)
    return ddim_latents


def null_optimization(whole_model, text_embeds, latents, num_ddim_steps, guidance_scale, num_inner_steps, epsilon, orig_dtype=None):
    uncond_embeddings, cond_embeddings = text_embeds.chunk(2)
    uncond_embeddings_list = []
    latent_cur = latents[-1]
    bar = tqdm(total=num_inner_steps * num_ddim_steps)
    for i in range(num_ddim_steps):
        uncond_embeddings = uncond_embeddings.clone().detach().requires_grad_(True)
        optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
        latent_prev = latents[len(latents) - i - 2]
        t = whole_model.scheduler.timesteps[i]
        with torch.no_grad():
            noise_pred_cond = get_noise_pred_single(whole_model, latent_cur, t, cond_embeddings)
        for j in range(num_inner_steps):
            with torch.enable_grad():
                noise_pred_uncond = get_noise_pred_single(whole_model, latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = inversion_prev_step(whole_model, noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_item = loss.item()
            bar.update()
            if loss_item < epsilon + i * 2e-5:
                break
        for j in range(j + 1, num_inner_steps):
            bar.update()
        uncond_embeddings_list.append(uncond_embeddings[:1].detach().to(orig_dtype))
        with torch.no_grad():
            context = torch.cat([uncond_embeddings, cond_embeddings])
            latent_cur = get_noise_pred(whole_model, text_embeds, latent_cur, t, guidance_scale, False, context)
    bar.close()
    return uncond_embeddings_list


def invert(whole_model, latent_gt, prompt, num_ddim_steps, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5,
           verbose=False, orig_dtype=None):
    text_embeds = init_prompt(whole_model, prompt, whole_model.unet.device, whole_model.unet.dtype)
    ptp_utils.register_attention_control(whole_model, None)
    if verbose:
        print("DDIM inversion...")
    ddim_latents = ddim_inversion(whole_model, text_embeds, latent_gt, num_ddim_steps)
    if verbose:
        print("Null-text optimization...")
    uncond_embeddings = null_optimization(whole_model, text_embeds, ddim_latents, num_ddim_steps, guidance_scale, num_inner_steps, early_stop_epsilon, orig_dtype=orig_dtype)
    #return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
    return ddim_latents[-1], uncond_embeddings



