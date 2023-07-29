# image_upscaling_pipeline.py

from diffusers import StableDiffusionLatentUpscalePipeline
import torch
import os
import typing
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

def add_noise(image, noise_scale):
    """
    Add noise to the input image.
    """
    image = TF.to_tensor(image)
    noise = torch.randn_like(image) * noise_scale
    noisy_image = torch.clamp(image + noise, 0, 1)
    return TF.to_pil_image(noisy_image)

def upscale_images_with_attention_slicing(slice_size: typing.Union[str, int, None] = 'auto', noise_scale: float = 0.1):
    model_id = "stabilityai/sd-x2-latent-upscaler"
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    upscaler.to("cuda")

    generator = torch.manual_seed(33)

    # Provide the path to the input images folder
    input_image_folder = "./output"

    # Create a folder to store the upscaled images
    if not os.path.exists('upscaled'):
        os.makedirs('upscaled')

    # Loop through all images in the input folder
    for filename in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, filename)

        # Load the image
        image = Image.open(image_path)

        # Resize the image to 512x512 (low-resolution)
        low_res_image = image.resize((512, 512))

        # Add noise to the low-resolution image
        low_res_image_with_noise = add_noise(low_res_image, noise_scale)

        # Upscale the low-resolution image using the upscaler
        upscaled_image = upscaler(
            prompt="dummy",  # Provide a dummy prompt for each image
            image=low_res_image_with_noise,
            num_inference_steps=50,  # Increase the inference steps for better quality
            guidance_scale=0,
            generator=generator,
        ).images[0]

        # Save the upscaled image
        output_path = os.path.join("upscaled", f"upscaled_{filename}")
        upscaled_image.save(output_path)

if __name__ == "__main__":
    upscale_images_with_attention_slicing(slice_size=64, noise_scale=0.1)  # You can adjust the noise_scale as desired
