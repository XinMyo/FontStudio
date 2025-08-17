import os
import torch
from SRM.SRM_SAET import SRM_SAET
from SRM.SRMpipeline import StableDiffusionXLPipeline
from SRM.SVD import SVDAutoencoderKL
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="SRM inference script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--svd_path",
        default="models/SVD_model",
        help="Path to the SVD model"
    )
    parser.add_argument(
        "--sd_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the Stable Diffusion base model"
    )
    parser.add_argument(
        "--font_root",
        default="font_sample",
        help="Root directory of fonts"
    )
    parser.add_argument(
        "--save_root",
        default="test",
        help="Directory to save results"
    )
    parser.add_argument(
        "--prompt",
        required=True,   
        help="Prompt for image generation"
    )

    return parser.parse_args()

args = get_args()

svd_path = args.svd_path
sd_path = args.sd_path
root = args.font_root
save_root = args.save_root
prompt = args.prompt

save_path=os.path.join(save_root,'image')
prior_path=os.path.join(save_root,'prior')
if not os.path.exists(save_path):
    os.makedirs(save_path)

svd = SVDAutoencoderKL.from_pretrained(svd_path, torch_dtype=torch.float16).to("cuda")
sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    sd_path,
    torch_dtype=torch.float16,
    variant="fp16",
    vae=svd,
).to("cuda")
SRM = SRM_SAET(pipe=sd_pipe, vae=svd)

ref_image=None

for i,ch in enumerate(os.listdir(root)):
    svd_mask = Image.open(os.path.join(root,ch)).resize((1024,1024)).convert('RGB')
    prior=Image.open(os.path.join(prior_path,ch))

    result_img = SRM(
        prompt=prompt,
        image=prior,
        num_inference_steps=30,
        noise_strength = 0.8,
        mask=svd_mask,
        ref_image=ref_image,
        SAET_steps=24
    )
    # Use first image as reference for subsequent iterations
    result_img[0].save(os.path.join(save_path,ch))
    if i==0:
        print('ref:',ch)
        ref_image=result_img[0]
