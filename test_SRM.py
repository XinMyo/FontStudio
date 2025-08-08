import os
import torch
from SRM.SRM_SAET import SRM_SAET
from SRM.SRMpipeline import StableDiffusionXLPipeline
from SRM.SVD import SVDAutoencoderKL
from PIL import Image


svd_path="your/path/to/svd-model"
sd_path="stabilityai/stable-diffusion-xl-base-1.0"

font_root='font_sample'
save_root='test'

prompt = "bamboo"
word='FontStudio'


root=os.path.join(font_root,word)
save_path=os.path.join(save_root,word,'image')
prior_path=os.path.join(save_root,word,'prior')
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