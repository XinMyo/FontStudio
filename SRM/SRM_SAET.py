from PIL import Image
from .SRM import SRM
from torchvision import transforms

# SAET is the only addition based on SRM
class SRM_SAET(SRM):
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        ref_image: Image.Image = None,
        noise_strength: float = 0.8,
        num_inference_steps: int = 30,
        SAET_steps: int = 25
    ):
        I_prime = self.black_background(image, mask)
        latents = self.encode_image(I_prime)
        latents_noisy, remaining_steps = self.add_noise(latents, noise_strength, num_inference_steps)
        # Encode reference image if provided
        if ref_image is not None:
            if ref_image.mode == 'RGBA':
                ref_image = ref_image.convert('RGB')
            self.reference_latents = self.encode_image(ref_image)
        else:
            self.reference_latents = None



        output = self.pipe(
            prompt=prompt,
            latents=latents_noisy,
            num_inference_steps=remaining_steps,
            mask=transforms.ToTensor()(mask).unsqueeze(0),
            reference_latents=self.reference_latents,
            SAET_steps=SAET_steps
        )
        return output.images