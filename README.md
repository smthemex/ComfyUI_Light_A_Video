# ComfyUI_Light_A_Video
[Light-A-Video](https://github.com/bcmi/Light-A-Video): Training-free Video Relighting via Progressive Light Fusion,you can use it in comfyUI


# Notice
* 底模是animatediff，所以只能跑512，需要等官方放出CogVideoX-2B的代码才能跑其他分辨率

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
    ├── animatediff-motion-adapter-v1-5-3.safetensors  # rename or not 或者随便换个名字
```
* if use [sam2](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt) to get mask image，如果使用sam2需要才下载模型，你使用BiRefNet是不用的（当然要下BiRefNet模型），注意sam2的注意点在正中，所以主体最好在中间。
```
--   ComfyUI/models/Light_A_Video
    ├── sam2_b.pt  #会自动下载
```  

# 4.Tips
* The prompt in the middle is used for the inner painting mode, and there is no need to fill in the light prompt, but the prompt related to the subject needs to be filled； 
* 中间的prompt是用于内绘模式的，无需填写灯光提示，需要填写主体相关的prompt；
* mask_repo：The method to get the mask is either to fill in 'ZhengPeng7/BiRefNet', or not to fill in, and it will automatically use sam2 or use the ‘mask_img’ interface to connect to the mask video；
* 获取mask的方法要么填'ZhengPeng7/BiRefNet'的repo或者本地绝对地址，要么不填，会自动用sam2模式，或者用mask_img接口连入mask视频；


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
