import os
import torch
import imageio
import argparse
from types import MethodType
import safetensors.torch as sf
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import MotionAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

from .src.ic_light import BGSource
from .src.ic_light import Relighter
from .src.animatediff_inpaint_pipe import AnimateDiffVideoToVideoPipeline
from .src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from .src.tools import read_video, read_mask,set_all_seed, get_fg_video




def main(args):
    
    config  = OmegaConf.load(args.config)
    device = torch.device('cuda')
    adopted_dtype = torch.float16
    set_all_seed(42)
    
    ## vdm model
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_model)
    #MotionAdapter.from_single_file(args.motion_adapter_model, torch_dtype=adopted_dtype)
    ## pipeline
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(args.sd_model, motion_adapter=adapter)
   
    eul_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        args.sd_model,
        subfolder="scheduler",
        beta_schedule="linear",
    )

    pipe.scheduler = eul_scheduler
    pipe.enable_vae_slicing()
    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    ## ic-light model
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
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
    sd_offset = sf.load_file(args.ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged
    text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    vae = vae.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # Consistent light attention
    @torch.inference_mode()
    def custom_forward_CLA(self, 
                        hidden_states, 
                        gamma=config.get("gamma", 0.5),
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
        
        # addition key and value
        mean_key = key.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True)
        mean_value = value.reshape(2,-1,shape[1],shape[2],shape[3]).mean(dim=1,keepdim=True)
        mean_key = mean_key.expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
        mean_value = mean_value.expand(-1,shape[0]//2,-1,-1,-1).reshape(shape[0],shape[1],shape[2],shape[3])
        add_hidden_state = F.scaled_dot_product_attention(query, mean_key, mean_value, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        # mix
        hidden_states = (1-gamma)*hidden_states + gamma*add_hidden_state
        
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

    ## consistency light attention
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
    ic_light_pipe = ic_light_pipe.to(device)
    
    #############################  params  ######################################
    strength = config.get("strength", 0.5)
    num_step = config.get("num_step", 50)
    text_guide_scale = config.get("text_guide_scale", 4)
    seed = config.get("seed")
    image_width = config.get("width", 512)
    image_height = config.get("height", 512)
    n_prompt = config.get("n_prompt", "")
    inpaint_prompt = config.get("inpaint_prompt", "")
    relight_prompt = config.get("relight_prompt", "")
    video_path = config.get("video_path", "")
    bg_source = BGSource[config.get("bg_source")]
    save_path = config.get("save_path")

    ##############################  infer  #####################################
    generator = torch.manual_seed(seed)
    video_name = os.path.basename(video_path)
    video_list, video_name = read_video(video_path, image_width, image_height)
    mask_folder = os.path.join("masks_animatediff", video_name.split('.')[-2])
    mask_list = read_mask(mask_folder)

    print("################## begin ##################")
    ## get foreground video
    fg_video_tensor = get_fg_video(video_list, mask_list, device, adopted_dtype) ## torch.Size([16, 3, 512, 512])

    with torch.no_grad():
        relighter = Relighter(
            pipeline=ic_light_pipe,
            relight_prompt=relight_prompt,
            bg_source=bg_source, 
            generator=generator,
            )
        vdm_init_latent = relighter(fg_video_tensor)

        ## infer
        num_inference_steps = num_step
        output = pipe(
            ic_light_pipe=ic_light_pipe,
            relight_prompt=relight_prompt,
            bg_source=bg_source,
            mask=mask_list,
            vdm_init_latent=vdm_init_latent,
            video=video_list,
            prompt=inpaint_prompt,
            strength=strength,
            negative_prompt=n_prompt,
            guidance_scale=text_guide_scale,
            num_inference_steps=num_inference_steps,
            height=image_height,
            width=image_width,
            generator=generator,
        )

        frames = output.frames[0]
        results_path = f"{save_path}/inpaint_{video_name}"
        imageio.mimwrite(results_path, frames, fps=8)
        print(f"relight with bg generation! prompt:{relight_prompt}, light:{bg_source.value}, save in {results_path}.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sd_model", type=str, default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument("--motion_adapter_model", type=str, default="guoyww/animatediff-motion-adapter-v1-5-3")
    parser.add_argument("--ic_light_model", type=str, default="./models/iclight_sd15_fc.safetensors")
    
    parser.add_argument("--config", type=str, default="configs/relight_inpaint/car.yaml", help="the config file for each sample.")
    
    args = parser.parse_args()
    main(args)