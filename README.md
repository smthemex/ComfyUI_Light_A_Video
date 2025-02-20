# ComfyUI_Light_A_Video
[Light-A-Video](https://github.com/bcmi/Light-A-Video): Training-free Video Relighting via Progressive Light Fusion,you can use it in comfyUI


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
* if use sam2 in inpaint mode,need 'imageio' and 'ultralytics>=8.3.0'

# 3.Model
* any sd1.5 checkpoints
```
--   ComfyUI/models/checkpoints
    ├── any sd1.5 checkpoints
```
* iclight_sd15_fc.safetensors from [here](https://huggingface.co/lllyasviel/ic-light/tree/main)
* animatediff-motion-adapter-v1-5-3.safetensors from [here](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/tree/main)
```
--   ComfyUI/models/controlnet
    ├── iclight_sd15_fc.safetensors
    ├── animatediff-motion-adapter-v1-5-3.safetensors  # rename or not 随便换个名字
```
* if use [sam2](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt)
```
--   ComfyUI/models/Light_A_Video
    ├── sam2_b.pt  #会自动下载
```  

# 4.Tips
* The middle prompts is only used for inpaint 中间的prompt是用于内绘模式的；
* mask_repo：use 'ZhengPeng7/BiRefNet' get mask，or keep it in empty to use sam2 to get mask 获取mask的方法要么填'ZhengPeng7/BiRefNet'，要么不填，会自动用sam2；


# 5.Example
* ic-light
![](https://github.com/smthemex/ComfyUI_Light_A_Video/blob/main/example_ic.png)
* inpanit
![ ](https://github.com/smthemex/ComfyUI_Light_A_Video/blob/main/example_in.png)


# Citation
```
@article{zhou2025light,
  title={Light-A-Video: Training-free Video Relighting via Progressive Light Fusion},
  author={Zhou, Yujie and Bu, Jiazi and Ling, Pengyang and Zhang, Pan and Wu, Tong and Huang, Qidong and Li, Jinsong and Dong, Xiaoyi and Zang, Yuhang and Cao, Yuhang and others},
  journal={arXiv preprint arXiv:2502.08590},
  year={2025}
}
```
