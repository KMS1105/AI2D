# sd_controlnet.py
# Core Stable Diffusion + ControlNet pipeline for img2img processing

import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image

class SDControlNetPipeline:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", controlnet_model_name=None, device="cuda"):
        self.device = device
        self.controlnet = ControlNetModel.from_pretrained(controlnet_model_name) if controlnet_model_name else None
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.safety_checker = lambda images, **kwargs: (images, False)  # Disable safety checker

    def img2img(self, init_image, prompt, strength=0.7, guidance_scale=7.5, controlnet_weight=1.0, num_inference_steps=20):
        """
        Apply img2img with ControlNet.
        """
        image = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_weight
        ).images[0]
        return image
