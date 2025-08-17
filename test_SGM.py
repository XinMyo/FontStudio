import os
import torch
from PIL import Image
from diffusers import ControlNetModel,  AutoencoderKL
from SGM.SGMpipeline import StableDiffusionXLControlNetPipeline
from torchvision.transforms import ToTensor
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="SGM inference script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base_model_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the base model"
    )
    parser.add_argument(
        "--controlnet_path",
        default="models/controlnet-model",
        help="Path to the ControlNet model"
    )
    parser.add_argument(
        "--vae_path",
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to the VAE model"
    )
    parser.add_argument(
        "--font_root",
        default="font_sample/FontStudio",
        help="Root directory of fonts"
    )
    parser.add_argument(
        "--save_root",
        default="test/FontStudio",
        help="Directory to save results"
    )
    parser.add_argument(
    "--prompt",
    required=True,   # 必填
    help="Prompt for image generation"
    )

    return parser.parse_args()

args = get_args()

base_model_path = args.base_model_path
controlnet_path = args.controlnet_path
vae_path = args.vae_path
root = args.font_root
save_root = args.save_root
prompt = args.prompt

save_path=os.path.join(save_root,'prior')
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

