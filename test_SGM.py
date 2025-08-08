import os
import torch
from PIL import Image
from diffusers import ControlNetModel,  AutoencoderKL
from SGM.SGMpipeline import StableDiffusionXLControlNetPipeline
from torchvision.transforms import ToTensor

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path = "your/path/to/controlnet-model"
vae_path="madebyollin/sdxl-vae-fp16-fix"

font_root='font_sample'
save_root='test'

prompt = "bamboo"
word='FontStudio'


root=os.path.join(font_root,word)
save_path=os.path.join(save_root,word,'prior')
if not os.path.exists(save_path):
    os.makedirs(save_path)

controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


controlnet_conditioning_scale = 1.0  

ref_image=None
ref_font=None



for i,ch in enumerate(os.listdir(root)):
    img = Image.open(os.path.join(root,ch)).convert('L')
    cross_attention_kwargs = {"ma_mask":ToTensor()(img.convert('L'))}
    if ref_image is not None:
        cross_attention_kwargs["ref_mask"]=ToTensor()(ref_image.convert('L'))
    prior = pipe(
        prompt, 
        image=img, 
        num_inference_steps=30,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        width=1024,
        height=1024,
        ref_image=ref_image,
        ref_font=ref_font,
        SAET_steps=100,
        cross_attention_kwargs = cross_attention_kwargs
    ).images
    
    prior[0].save(os.path.join(save_path,ch))
    # Use first image as reference for subsequent iterations
    if i==0:
        print('ref:',ch)
        ref_image=img
        ref_font=prior 

