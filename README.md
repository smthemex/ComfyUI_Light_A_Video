# ComfyUI_Light_A_Video
[Light-A-Video](https://github.com/bcmi/Light-A-Video): Training-free Video Relighting via Progressive Light Fusion,you can use it in comfyUI


# Update
* support ‘cogvideox’ and ‘wan2.1 diffusers’  视频底模同步官方代码,支持cogvideox 和 ‘万相2.1 diffusers’
* wan2.1 image size  set to 832 * 480, cogvideox image size  set to 720 * 480 万相的图片尺寸设置832 * 480，cog设置720 * 480
* wan2.1 or cogvideoX need 4090 or more Vram，目前还没有优化，主要是wan的T5太大了

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Light_A_Video.git
```
---

# 2. Requirements  
```
pip install -r requirements.txt
```
* if use sam2  to get mask image (use inpaint mode),need 'ultralytics>=8.3.51',使用sam2模式获取mask图片时（内绘模式），需要'ultralytics>=8.3.51，可能低一两个版本也能用，不测试了。
  
* wan2.1 need diffusers main ,so install it as below/万相需要的diffuser版本太新，需要按以下方法安装:
```
pip install git+https://github.com/huggingface/diffusers

or

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```
# 3.Model
**3.1 base models**
* any sd1.5 checkpoints
```
--   ComfyUI/models/checkpoints
    ├── any sd1.5 checkpoints
```
* iclight_sd15_fc.safetensors from [here](https://huggingface.co/lllyasviel/ic-light/tree/main)
```
--   ComfyUI/models/controlnet
    ├── iclight_sd15_fc.safetensors
```
**3.2 use animatediff**
* animatediff-motion-adapter-v1-5-3.safetensors from [here](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/tree/main)
```
--   ComfyUI/models/controlnet
    ├── animatediff-motion-adapter-v1-5-3.safetensors  # rename or not 或者随便换个名字
```
* if use [sam2](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt) to get mask image，如果使用sam2需要才下载模型，你使用BiRefNet是不用的（当然要下BiRefNet模型），注意sam2的注意点在正中，所以主体最好在中间。
```
--   ComfyUI/models/Light_A_Video
    ├── sam2_b.pt  #会自动下载
```
**3.3 use wan2.1**
* fill repo [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) or local repo (使用抱脸的repo在线下载或者预下载存放在本地的本地repo地址)

**3.4 use cogvideoX**
* fill repo [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b/tree/main)  or local repo (使用抱脸的repo在线下载或者预下载存放在本地的本地repo地址)

# 4.Tips
* 第三条为cog和wan专用prompt，只需要填写主体，比如一只熊什么的，不要填写灯光（The third is a special prompt for cog and wan, only need to fill in the main body, such as a bear or something, do not fill in the light）;  
* The prompt in the middle is used for the inner painting mode, and there is no need to fill in the light prompt, but the prompt related to the subject needs to be filled； 
* 中间的prompt是用于内绘模式的，无需填写灯光提示，需要填写主体相关的prompt；
* mask_repo：The method to get the mask is either to fill in 'ZhengPeng7/BiRefNet', or not to fill in, and it will automatically use sam2 or use the ‘mask_img’ interface to connect to the mask video；
* 获取mask的方法要么填'ZhengPeng7/BiRefNet'的repo或者本地绝对地址，要么不填，会自动用sam2模式，或者用mask_img接口连入mask视频；


# 5.Example
* wan2.1 ic-light
![](https://github.com/smthemex/ComfyUI_Light_A_Video/blob/main/assets/example_w.png)
* animatediff ic-light
![](https://github.com/smthemex/ComfyUI_Light_A_Video/blob/main/assets/example_ic.png)
* animatediff inpanit
![](https://github.com/smthemex/ComfyUI_Light_A_Video/blob/main/assets/example_in.png)


# Citation
```
@article{zhou2025light,
  title={Light-A-Video: Training-free Video Relighting via Progressive Light Fusion},
  author={Zhou, Yujie and Bu, Jiazi and Ling, Pengyang and Zhang, Pan and Wu, Tong and Huang, Qidong and Li, Jinsong and Dong, Xiaoyi and Zang, Yuhang and Cao, Yuhang and others},
  journal={arXiv preprint arXiv:2502.08590},
  year={2025}
}
```
