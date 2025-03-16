from PIL import Image,ImageSequence
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import os
import imageio
import random
from diffusers.utils import  export_to_video

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def numpy2pytorch(imgs, device, dtype):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h.to(device=device, dtype=dtype)

def get_fg_video(video_list, mask_list, device, dtype):
    video_np = np.stack(video_list, axis=0)
    mask_np = np.stack(mask_list, axis=0)
    mask_bool = mask_np == 255
    video_fg = np.where(mask_bool, video_np, 127)

    h = torch.from_numpy(video_fg).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h.to(device=device, dtype=dtype)


def pad(x, p, i):
    return x[:i] if len(x) >= i else x + [p] * (i - len(x))

def gif_to_mp4(gif_path, mp4_path):
    clip = VideoFileClip(gif_path)
    clip.write_videofile(mp4_path)

def generate_light_sequence(light_tensor, num_frames=16, direction="r"):

    if direction in "l":
        target_tensor = torch.rot90(light_tensor, k=1, dims=(2, 3))
    elif direction in "r":
        target_tensor = torch.rot90(light_tensor, k=-1, dims=(2, 3))
    else:
        raise ValueError("direction must be either 'r' for right or 'l' for left")
        
    # Generate the sequence
    out_list = []
    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1) 
        interpolated_matrix = (1 - t) * light_tensor + t * target_tensor
        out_list.append(interpolated_matrix)
    
    out_tensor = torch.stack(out_list, dim=0).squeeze(1)

    return out_tensor

def tensor2vid(video: torch.Tensor, processor, output_type="np"):

    batch_size, channels, num_frames, height, width = video.shape ## [1, 4, 16, 512, 512]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

def read_video(video_path:str, image_width, image_height):
    extension = video_path.split('.')[-1].lower()
    video_name = os.path.basename(video_path)
    video_list = []

    if extension in "gif":
        ## input from gif
        video = Image.open(video_path)
        for i, frame in enumerate(ImageSequence.Iterator(video)):
            frame = np.array(frame.convert("RGB"))
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    elif extension in "mp4":
        ## input from mp4
        reader = imageio.get_reader(video_path)
        for frame in reader:
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    else:
        raise ValueError('Wrong input type')
    
    video_list = [Image.fromarray(frame) for frame in video_list]

    return video_list, video_name

def read_mask(mask_folder:str):
    mask_files = os.listdir(mask_folder)
    mask_files = sorted(mask_files)
    mask_list = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = Image.open(mask_path).convert('RGB')
        mask_list.append(mask)
    
    return mask_list

def decode_latents(vae, latents, decode_chunk_size: int = 16):
    
    latents = 1 / vae.config.scaling_factor * latents
    video = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        batch_latents = latents[i : i + decode_chunk_size]
        batch_latents = vae.decode(batch_latents).sample
        video.append(batch_latents)

    video = torch.cat(video)

    return video

def encode_video(vae, video, decode_chunk_size: int = 16) -> torch.Tensor:
    latents = []
    for i in range(0, len(video), decode_chunk_size):
        batch_video = video[i : i + decode_chunk_size]
        batch_video = vae.encode(batch_video).latent_dist.mode()
        latents.append(batch_video)
    return torch.cat(latents)

def vis_video(input_video, video_processor, save_path):
    ## shape: 1, c, f, h, w
    relight_video = video_processor.postprocess_video(video=input_video, output_type="pil")
    export_to_video(relight_video[0], save_path)
    
def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True