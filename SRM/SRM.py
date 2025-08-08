import torch
from diffusers import  DDIMScheduler
from PIL import Image
from .SVD import SVDAutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from .timesteps import get_timesteps
from .SRMpipeline import StableDiffusionXLPipeline
from torchvision import transforms

class SRM:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
        vae: SVDAutoencoderKL = None,
        image_processor: VaeImageProcessor = None,
    ):
        self.pipe = pipe
        self.device = next(self.pipe.unet.parameters()).device
        self.pipe.vae=vae
        self.vae = vae if vae is not None else pipe.vae
        self.vae.eval()

        self.image_processor = image_processor if image_processor is not None else VaeImageProcessor(vae_scale_factor=8)

        # Replace default scheduler with DDIM for flexible intermediate timestep sampling
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    # Convert masked region to black background
    def black_background(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGBA")
        mask = mask.convert("L").resize(image.size)
        black_bg = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image.putalpha(mask)
        black_bg.paste(image, (0, 0), image)
        return black_bg.convert("RGB")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        tensor = self.image_processor.preprocess(image=image).to(self.device, dtype=torch.float16)
        with torch.no_grad():
            latents = self.vae.encode(tensor).latent_dist.sample() * 0.18215
        return latents

    def add_noise(self, latents: torch.Tensor, noise_strength: float, num_inference_steps: int) -> tuple[torch.Tensor, int]:
        
        timestep_idx = int((1-noise_strength) * len(self.pipe.scheduler.timesteps))
        timestep_idx = min(timestep_idx, len(self.pipe.scheduler.timesteps) - 1)
        t = self.pipe.scheduler.timesteps[timestep_idx]

        self.timesteps = self.pipe.scheduler.timesteps
        # Override retrieve_timesteps to support mid-step inference
        self.pipe.retrieve_timesteps=get_timesteps(noise_strength)

        noise = torch.randn_like(latents)
        latents_noisy = self.pipe.scheduler.add_noise(latents, noise, t)
        return latents_noisy,  num_inference_steps

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        noise_strength: float = 0.8,
        num_inference_steps: int = 50,
    ):
        I_prime = self.black_background(image, mask)
        # Encode masked image to latents
        latents = self.encode_image(I_prime)
        latents_noisy, remaining_steps = self.add_noise(latents, noise_strength, num_inference_steps)

        output = self.pipe(
            prompt=prompt,
            latents=latents_noisy,
            num_inference_steps=remaining_steps,
            mask=transforms.ToTensor()(mask).unsqueeze(0)
        )
        return output.images
