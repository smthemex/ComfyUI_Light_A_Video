# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from diffusers import StableDiffusionPipeline
from .lav_relight import load_ic_light_model,infer_relight
from .lav_wan_relight import load_ic_light_wan,infer_relight_wan
from .lav_cog_relight import load_ic_light_cog,infer_relight_cog
from .node_utils import load_images,tensor2pil_list
import folder_paths
from .src.ic_light import BGSource
from .src.tools import  set_all_seed

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
Light_A_Video_weigths_path = os.path.join(folder_paths.models_dir, "Light_A_Video")
if not os.path.exists(Light_A_Video_weigths_path):
    os.makedirs(Light_A_Video_weigths_path)
folder_paths.add_model_folder_path("Light_A_Video", Light_A_Video_weigths_path)


class Light_A_Video_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo": ("STRING", {"default":"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},),
                "model": (folder_paths.get_filename_list("checkpoints"),),
                "motion_adapter_model": (["none"] + folder_paths.get_filename_list("controlnet"),),
                "ic_light_model": (["none"] + folder_paths.get_filename_list("controlnet"),),
                "mode":(["relight","inpaint"],),
            },
        }

    RETURN_TYPES = ("MODEL_Light_A_Video",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "Light_A_Video"

    def loader_main(self, repo,model,motion_adapter_model, ic_light_model,mode):

        adopted_dtype = torch.float16
        # ic light
        sd_repo = os.path.join(current_node_path, "sd_repo")
        original_config_file=os.path.join(sd_repo,"v1-inference.yaml") #fix for desktop comfyUI
        if model!="none":
                ckpt_path=folder_paths.get_full_path("checkpoints",model)
        else:
            raise "no sd1.5 checkpoint"
        try:
            sd_pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,config=sd_repo, original_config=original_config_file)
        except:
            sd_pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path, config=sd_repo,original_config_file=original_config_file)
        ic_light_model=folder_paths.get_full_path("controlnet",ic_light_model)

        # video model
        if repo:
            if "wan" in repo.lower():
                print("***********Load wan diffuser ***********")
                pipe,ic_light_pipe=load_ic_light_wan(repo,sd_pipe,sd_repo,ckpt_path,ic_light_model,device,adopted_dtype)
                video_mode="wan"
            elif "cog" in repo.lower():
                print("***********Load cogvideox diffuser ***********")
                pipe,ic_light_pipe=load_ic_light_cog(repo,sd_pipe,sd_repo,ckpt_path,ic_light_model,device,adopted_dtype)
                video_mode="cog"
            else:
                raise "no string match wan or cog,check your repo name"
        else:
            # load animatediff model
            motion_repo=os.path.join(current_node_path,"animate_repo")
            print("***********Load animatediff model ***********")
            motion_adapter_model=folder_paths.get_full_path("controlnet",motion_adapter_model)
            pipe,ic_light_pipe=load_ic_light_model(sd_pipe,ic_light_model,ckpt_path,sd_repo,motion_repo,motion_adapter_model,device,adopted_dtype,mode)
            video_mode="animate"
        print("***********Load model done ***********")
        gc.collect()
        torch.cuda.empty_cache()
        return ({"model":pipe,"ic_light_pipe":ic_light_pipe,"mode":mode,"adopted_dtype":adopted_dtype,"video_mode":video_mode},)


class Light_A_Video_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_Light_A_Video",),
                "images": ("IMAGE",),
                "relight_prompt": ("STRING", {"default": "a bear walking on the rock, nature lighting, soft light", "multiline": True}),
                "vdm_prompt": ("STRING", {"default": "a bear walking on the rock", "multiline": True}),
                "inpaint_prompt": ("STRING", {"default": "a car driving on the beach, sunset over sea", "multiline": True}),
                "n_prompt": ("STRING", {"default": "bad quality, worse quality", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "num_step": ("INT", {"default": 25, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text_guide_scale": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16, "display": "number"}),
                "num_frames": ("INT", {"default": 49, "min": 10, "max": 1024, "step": 1, "display": "number"}),
                "bg_target": (["LEFT",  "RIGHT", "TOP", "BOTTOM","NONE",],),
                "mask_repo": ("STRING", {"default": "ZhengPeng7/BiRefNet"},),},
            "optional": {"mask_img": ("IMAGE",),
                         "fps": ("FLOAT", {"default": 8.0, "min": 8.0, "max": 100.0, "step": 0.1}),
                         },
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "Light_A_Video"

    def sampler_main(self, model,images,relight_prompt,vdm_prompt,inpaint_prompt, n_prompt,seed, num_step, strength,text_guide_scale,width, height,num_frames,bg_target,mask_repo,**kwargs):
        set_all_seed(42)

        local_sam=os.path.join(Light_A_Video_weigths_path,"sam2_b.pt")
        if not os.path.exists(local_sam):
            local_sam="sam2_b.pt"
        ref_image_list=tensor2pil_list(images,width,height)

        if bg_target=="NONE":
            bg_source = BGSource.NONE
        elif bg_target=="LEFT":
            bg_source = BGSource.LEFT
        elif bg_target=="RIGHT":
            bg_source = BGSource.RIGHT
        elif bg_target=="TOP":
            bg_source = BGSource.TOP
        else:
            bg_source = BGSource.BOTTOM

        mask_img=kwargs.get("mask_img",None)
        fps=kwargs.get("fps",8.0)
        ic_light_pipe=model.get("ic_light_pipe")
        pipe=model.get("model")
        video_mode=model.get("video_mode")
        mode=model.get("mode")
        adopted_dtype=model.get("adopted_dtype")

        if isinstance(mask_img,torch.Tensor): 
            mask_list=tensor2pil_list(mask_img,width,height)
            # for i,mask in enumerate(mask_list):
            #     mask.save(f"mask{i}.png")
            if len(mask_list)==1:
                mask_list=mask_list*len(ref_image_list)
                print("not enough mask, repeat mask to match the number of images")
        else:
            mask_list=None
        
        print("***********Start infer  ***********")
        if video_mode=="wan":
            iamge = infer_relight_wan(ic_light_pipe,pipe,strength,num_step,text_guide_scale,seed,width,height,n_prompt,relight_prompt,vdm_prompt,ref_image_list,bg_source,num_frames,mode,mask_list,device,adopted_dtype,mask_repo,fps,local_sam)
        elif video_mode=="cog":
            iamge = infer_relight_cog(ic_light_pipe,pipe,strength,num_step,text_guide_scale,seed,width,height,n_prompt,relight_prompt,vdm_prompt,ref_image_list,bg_source,num_frames,mode,mask_list,device,adopted_dtype,mask_repo,fps,local_sam)
        else:
            iamge = infer_relight(ic_light_pipe,pipe,strength,num_step,text_guide_scale,seed,width,height,n_prompt,relight_prompt,inpaint_prompt,ref_image_list,bg_source,mode,mask_list,device,adopted_dtype,mask_repo,fps,local_sam)
            
        torch.cuda.empty_cache()

        return (load_images(iamge),)


NODE_CLASS_MAPPINGS = {
    "Light_A_Video_Loader": Light_A_Video_Loader,
    "Light_A_Video_Sampler": Light_A_Video_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light_A_Video_Loader": "Light_A_Video_Loader",
    "Light_A_Video_Sampler": "Light_A_Video_Sampler",
}
