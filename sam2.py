import os
import torch
#import argparse
import numpy as np
from PIL import Image
import random
import imageio
import folder_paths


def get_mask(video_list,fps,local_sam, x=255, y=255):
    from ultralytics.models.sam import SAM2VideoPredictor #ultralytics>=8.3.0
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=local_sam)
    predictor = SAM2VideoPredictor(overrides=overrides)
    file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
    frames = [np.array(img) for img in video_list]
    video_file = os.path.join(folder_paths.get_input_directory(), f"audio_{file_prefix}_temp.mp4")
    imageio.mimsave(video_file, frames, fps=fps, codec='libx264')
    results = predictor(source=video_file,points=[x, y],labels=[1])
    mask_list=[]
    for i in range(len(results)):
        mask = (results[i].masks.data).squeeze().to(torch.float16)
        mask = (mask * 255).cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask).convert('RGB')
        mask_list.append(mask_image)
        # mask_dir = f'masks_animatediff/{video_name}'
        # if not os.path.exists(mask_dir):  
        #     os.makedirs(mask_dir)        
        mask_image.save(folder_paths.get_input_directory() + f'/{str(i).zfill(3)}.png')
    return mask_list


