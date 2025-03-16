import os
import torch
import imageio
#import argparse
import numpy as np
import safetensors.torch as sf

from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.hub import download_url_to_file
import gc
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

from .src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from .src.wan_pipe import WanVideoToVideoPipeline
from .src.ic_light import BGSource
from .utils.tools import set_all_seed, read_video


def load_ic_light_wan(repo,sd_pipe,sd_repo,ckpt_path,ic_light_model,device,adopted_dtype):
    # config  = OmegaConf.load(args.config)
    # device = torch.device('cuda')
    # adopted_dtype = torch.float16
    #set_all_seed(42)
    ## vdm model        
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=adopted_dtype)
    pipe = WanVideoToVideoPipeline.from_pretrained(repo, vae=vae, torch_dtype=adopted_dtype)

    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    ## module
    tokenizer = CLIPTokenizer.from_pretrained(sd_repo, subfolder="tokenizer")
    unet=sd_pipe.unet #UNet2DConditionModel.from_pretrained(sd_repo, subfolder="unet") 确认是否有效
    text_encoder = sd_pipe.text_encoder
    vae = sd_pipe.vae
    del sd_pipe
    gc.collect()


    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_() #torch.Size([320, 8, 3, 3])
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    unet_original_forward = unet.forward
    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    unet.forward = hooked_unet_forward

    ## ic-light model loader
    if not os.path.exists(ic_light_model):
        download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', 
                             dst=ic_light_model) #TO DO: change to ic-light model
    sd_offset = sf.load_file(ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged
    text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    vae = vae.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    @torch.inference_mode()
    def custom_forward_CLA(self, 
                        hidden_states, 
                        gamma=0.7,
                        encoder_hidden_states=None,
                        attention_mask=None, 
                        cross_attention_kwargs=None
                        ):

        batch_size, sequence_length, channel = hidden_states.shape

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)   
        value = self.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        shape = query.shape
        mean_key = key.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True).expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
        mean_value = value.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True).expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
        hidden_states_mean = F.scaled_dot_product_attention(query, mean_key, mean_value, attn_mask=None, dropout_p=0.0, is_causal=False)

        hidden_states = (1-gamma)*hidden_states + gamma*hidden_states_mean
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor





        return hidden_states
    
    ### attention
    from types import MethodType
    @torch.inference_mode()
    def prep_unet_self_attention(unet):

        for name, module in unet.named_modules():
            module_name = type(module).__name__
        
            name_split_list = name.split(".")
            cond_1 = name_split_list[0] in "up_blocks"
            cond_2 = name_split_list[-1] in ('attn1')
            
            if "Attention" in module_name and cond_1 and cond_2:
                cond_3 = name_split_list[1] 
                if cond_3 not in "3":
                    module.forward = MethodType(custom_forward_CLA, module)

        return unet

    ## attn module
    unet = prep_unet_self_attention(unet)
    
    ## ic-light-scheduler
    ic_light_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )
    ic_light_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=ic_light_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
    ic_light_pipe = ic_light_pipe.to(device=device, dtype=adopted_dtype)
    ic_light_pipe.vae.requires_grad_(False)
    ic_light_pipe.unet.requires_grad_(False)

    # pipeline = {
    #     "transformer": pipe.transformer,
    #     "text_encoder": pipe.text_encoder,
    #     "vae": pipe.vae
    # }


    return pipe,ic_light_pipe


def infer_relight_wan(ic_light_pipe,pipe,strength,num_step,text_guide_scale,seed,image_width,image_height,n_prompt,relight_prompt,vdm_prompt,video_list,bg_source,num_frames,mode,mask_list,device,adopted_dtype,repo,fps,local_sam):
    ##############################  infer  #####################################
    generator = torch.manual_seed(seed)
    # video_name = os.path.basename(video_path)
    # video_list, video_name = read_video(video_path, image_width, image_height)
    print("################## begin ##################")
    with torch.no_grad():
        num_inference_steps = int(round(num_step / strength))
        output = pipe(
            ic_light_pipe=ic_light_pipe,
            relight_prompt=relight_prompt,
            bg_source=bg_source,
            video=video_list,
            prompt=vdm_prompt, 
            negative_prompt=n_prompt,
            strength=strength, 
            guidance_scale=text_guide_scale, 
            num_inference_steps=num_inference_steps,
            height=image_height,
            num_frames=num_frames,
            width=image_width,
            generator=generator,
        )

        frames = output.frames[0]
        # frames = (frames * 255).astype(np.uint8)
        # results_path = f"{save_path}/relight_{video_name}"
        # imageio.mimwrite(results_path, frames, fps=14)
        # print(f"relight! prompt:{relight_prompt}, light:{bg_source.value}, save in {results_path}.")
    return frames


#def main(args):

    # config  = OmegaConf.load(args.config)
    # device = torch.device('cuda')
    # adopted_dtype = torch.float16
    # set_all_seed(42)

    # ## vdm model        
    # vae = AutoencoderKLWan.from_pretrained(args.vdm_model, subfolder="vae", torch_dtype=adopted_dtype)
    # pipe = WanVideoToVideoPipeline.from_pretrained(args.vdm_model, vae=vae, torch_dtype=adopted_dtype)

    # pipe = pipe.to(device=device, dtype=adopted_dtype)
    # pipe.vae.requires_grad_(False)
    # pipe.transformer.requires_grad_(False)

    # ## module
    # tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    # with torch.no_grad():
    #     new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    #     new_conv_in.weight.zero_() #torch.Size([320, 8, 3, 3])
    #     new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    #     new_conv_in.bias = unet.conv_in.bias
    #     unet.conv_in = new_conv_in
    # unet_original_forward = unet.forward

    # def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    #     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    #     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    #     new_sample = torch.cat([sample, c_concat], dim=1)
    #     kwargs['cross_attention_kwargs'] = {}
    #     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    # unet.forward = hooked_unet_forward

    # ## ic-light model loader
    # if not os.path.exists(args.ic_light_model):
    #     download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', 
    #                          dst=args.ic_light_model)
    # sd_offset = sf.load_file(args.ic_light_model)
    # sd_origin = unet.state_dict()
    # sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    # unet.load_state_dict(sd_merged, strict=True)
    # del sd_offset, sd_origin, sd_merged
    # text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    # vae = vae.to(device=device, dtype=adopted_dtype)
    # unet = unet.to(device=device, dtype=adopted_dtype)
    # unet.set_attn_processor(AttnProcessor2_0())
    # vae.set_attn_processor(AttnProcessor2_0())

    # Consistent light attention
    # @torch.inference_mode()
    # def custom_forward_CLA(self, 
    #                     hidden_states, 
    #                     gamma=0.7,
    #                     encoder_hidden_states=None,
    #                     attention_mask=None, 
    #                     cross_attention_kwargs=None
    #                     ):

    #     batch_size, sequence_length, channel = hidden_states.shape

    #     residual = hidden_states
    #     input_ndim = hidden_states.ndim
    #     if input_ndim == 4:
    #         batch_size, channel, height, width = hidden_states.shape
    #         hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    #     if attention_mask is not None:
    #         if attention_mask.shape[-1] != query.shape[1]:
    #             target_length = query.shape[1]
    #             attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
    #             attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
    #     if self.group_norm is not None:
    #         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    #     if encoder_hidden_states is None:
    #         encoder_hidden_states = hidden_states

    #     query = self.to_q(hidden_states)
    #     key = self.to_k(encoder_hidden_states)   
    #     value = self.to_v(encoder_hidden_states)
    #     inner_dim = key.shape[-1]
    #     head_dim = inner_dim // self.heads
    #     query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    #     key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    #     value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    #     hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
    #     shape = query.shape
    #     mean_key = key.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True).expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
    #     mean_value = value.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True).expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
    #     hidden_states_mean = F.scaled_dot_product_attention(query, mean_key, mean_value, attn_mask=None, dropout_p=0.0, is_causal=False)

    #     hidden_states = (1-gamma)*hidden_states + gamma*hidden_states_mean
    #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
    #     hidden_states = hidden_states.to(query.dtype)

    #     hidden_states = self.to_out[0](hidden_states)
    #     hidden_states = self.to_out[1](hidden_states)

    #     if input_ndim == 4:
    #         hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    #     if self.residual_connection:
    #         hidden_states = hidden_states + residual

    #     hidden_states = hidden_states / self.rescale_output_factor


    #     return hidden_states

    # ### attention
    # from types import MethodType
    # @torch.inference_mode()
    # def prep_unet_self_attention(unet):

    #     for name, module in unet.named_modules():
    #         module_name = type(module).__name__
        
    #         name_split_list = name.split(".")
    #         cond_1 = name_split_list[0] in "up_blocks"
    #         cond_2 = name_split_list[-1] in ('attn1')
            
    #         if "Attention" in module_name and cond_1 and cond_2:
    #             cond_3 = name_split_list[1] 
    #             if cond_3 not in "3":
    #                 module.forward = MethodType(custom_forward_CLA, module)

    #     return unet

    # ## attn module
    # unet = prep_unet_self_attention(unet)

    # ## ic-light-scheduler
    # ic_light_scheduler = DPMSolverMultistepScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.012,
    #     algorithm_type="sde-dpmsolver++",
    #     use_karras_sigmas=True,
    #     steps_offset=1
    # )
    # ic_light_pipe = StableDiffusionImg2ImgPipeline(
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=unet,
    #     scheduler=ic_light_scheduler,
    #     safety_checker=None,
    #     requires_safety_checker=False,
    #     feature_extractor=None,
    #     image_encoder=None
    # )
    # ic_light_pipe = ic_light_pipe.to(device=device, dtype=adopted_dtype)
    # ic_light_pipe.vae.requires_grad_(False)
    # ic_light_pipe.unet.requires_grad_(False)

    #############################  params  ######################################
    # strength = config.get("strength", 0.4)
    # num_step = config.get("num_step", 25)
    # text_guide_scale = config.get("text_guide_scale", 2)
    # seed = config.get("seed")
    # image_width = config.get("width", 512)
    # image_height = config.get("height", 512)
    # negative_prompt = config.get("n_prompt", "")
    # vdm_prompt = config.get("vdm_prompt", "")
    # relight_prompt = config.get("relight_prompt", "")
    # video_path = config.get("video_path", "")
    # bg_source = BGSource[config.get("bg_source")]
    # save_path = config.get("save_path")
    # num_frames = config.get("num_frames", 49)
    
    # ##############################  infer  #####################################
    # generator = torch.manual_seed(seed)
    # video_name = os.path.basename(video_path)
    # video_list, video_name = read_video(video_path, image_width, image_height)

    # print("################## begin ##################")
    # with torch.no_grad():
    #     num_inference_steps = int(round(num_step / strength))
        
    #     output = pipe(
    #         ic_light_pipe=ic_light_pipe,
    #         relight_prompt=relight_prompt,
    #         bg_source=bg_source,
    #         video=video_list,
    #         prompt=vdm_prompt, 
    #         negative_prompt=negative_prompt,
    #         strength=strength, 
    #         guidance_scale=text_guide_scale, 
    #         num_inference_steps=num_inference_steps,
    #         height=image_height,
    #         num_frames=num_frames,
    #         width=image_width,
    #         generator=generator,
    #     )

    #     frames = output.frames[0]
    #     frames = (frames * 255).astype(np.uint8)
    #     results_path = f"{save_path}/relight_{video_name}"
    #     imageio.mimwrite(results_path, frames, fps=14)
    #     print(f"relight! prompt:{relight_prompt}, light:{bg_source.value}, save in {results_path}.")
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--sd_model", type=str, default="stablediffusionapi/realistic-vision-v51")
#     parser.add_argument("--vdm_model", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
#     parser.add_argument("--ic_light_model", type=str, default="./models/iclight_sd15_fc.safetensors")
    
#     parser.add_argument("--config", type=str, default="configs/wan_relight/man.yaml", help="the config file for each sample.")
    
#     args = parser.parse_args()
#     main(args)
