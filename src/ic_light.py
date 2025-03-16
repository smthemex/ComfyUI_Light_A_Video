import torch
import numpy as np
from enum import Enum
import math

import torch.nn.functional as F
from .tools import resize_and_center_crop, numpy2pytorch, pad, decode_latents, encode_video

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

class Relighter:
    def __init__(self, 
                 pipeline, 
                 relight_prompt="",
                 num_frames=16,
                 image_width=512,
                 image_height=512, 
                 num_samples=1, 
                 steps=15, 
                 cfg=2, 
                 lowres_denoise=0.9, 
                 bg_source=BGSource.RIGHT, 
                 generator=None,
                 ):
        
        self.pipeline = pipeline
        self.image_width = image_width
        self.image_height = image_height
        self.num_samples = num_samples
        self.steps = steps
        self.cfg = cfg
        self.lowres_denoise = lowres_denoise
        self.bg_source = bg_source
        self.generator = generator
        self.device = pipeline.device
        self.num_frames = num_frames
        self.vae = self.pipeline.vae
        
        self.a_prompt = "best quality"
        self.n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        positive_prompt = relight_prompt + ', ' + self.a_prompt
        negative_prompt = self.n_prompt
        tokenizer = self.pipeline.tokenizer
        device = self.pipeline.device
        vae = self.vae
        
        conds, unconds = self.encode_prompt_pair(tokenizer, device, positive_prompt, negative_prompt)
        input_bg = self.create_background()
        # bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)
        # bg_latent = numpy2pytorch([bg], device, vae.dtype)
        # bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        if self.bg_source == BGSource.NONE:
             shape = (1, 4, self.image_width//8, self.image_height//8)
             bg_latent = torch.randn(shape, generator=generator, device=device, dtype=vae.dtype)
        else:
             bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)
             bg_latent = numpy2pytorch([bg], device, vae.dtype)
             bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        self.bg_latent = bg_latent.repeat(self.num_frames, 1, 1, 1) ## 固定光源
        self.conds = conds.repeat(self.num_frames, 1, 1)
        self.unconds = unconds.repeat(self.num_frames, 1, 1)
        
    def encode_prompt_inner(self, tokenizer, txt):
        max_length = tokenizer.model_max_length
        chunk_length = tokenizer.model_max_length - 2
        id_start = tokenizer.bos_token_id
        id_end = tokenizer.eos_token_id
        id_pad = id_end

        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.pipeline.text_encoder(token_ids).last_hidden_state
        return conds

    def encode_prompt_pair(self, tokenizer, device, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(tokenizer, positive_prompt)
        uc = self.encode_prompt_inner(tokenizer, negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c.to(device), uc.to(device)

    def create_background(self):
        
        max_pix = 255
        min_pix = 0
        
        print(f"max light pix:{max_pix}, min light pix:{min_pix}")
        
        if self.bg_source == BGSource.NONE:
            return None
        elif self.bg_source == BGSource.LEFT:
            gradient = np.linspace(max_pix, min_pix, self.image_width)
            image = np.tile(gradient, (self.image_height, 1))
            return np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif self.bg_source == BGSource.RIGHT:
            gradient = np.linspace(min_pix, max_pix, self.image_width)
            image = np.tile(gradient, (self.image_height, 1))
            return np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif self.bg_source == BGSource.TOP:
            gradient = np.linspace(max_pix, min_pix, self.image_height)[:, None]
            image = np.tile(gradient, (1, self.image_width))
            return np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif self.bg_source == BGSource.BOTTOM:
            gradient = np.linspace(min_pix, max_pix, self.image_height)[:, None]
            image = np.tile(gradient, (1, self.image_width))
            return np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise ValueError('Wrong initial latent!')
    
    @torch.no_grad()
    def __call__(self, input_video, init_latent=None, input_strength=None):
        input_latent = encode_video(self.vae, input_video)* self.vae.config.scaling_factor
        
        if input_strength:
            light_strength = input_strength
        else:
            light_strength = self.lowres_denoise

        if not init_latent:
            init_latent = self.bg_latent

        latents = self.pipeline(
            image=init_latent,
            strength=light_strength,
            prompt_embeds=self.conds,
            negative_prompt_embeds=self.unconds,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=int(round(self.steps / self.lowres_denoise)),
            num_images_per_prompt=self.num_samples,
            generator=self.generator,
            output_type='latent',
            guidance_scale=self.cfg,
            cross_attention_kwargs={'concat_conds': input_latent},
        ).images.to(self.pipeline.vae.dtype)

        relight_video = decode_latents(self.vae, latents)
        return relight_video