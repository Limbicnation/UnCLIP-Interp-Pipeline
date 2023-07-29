# Original script by the Hugging Faces Diffusers library

import argparse
import torch
import os
from diffusers import UnCLIPImageVariationPipeline
from PIL import Image

def main(interpolation_factor, steps):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

    # Initialize the pipeline with custom_args
    pipe = UnCLIPImageVariationPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha-image-variations",
        torch_dtype=dtype,
        interpolation_factor=interpolation_factor,
    )

    pipe.to(device)

    # Define local image paths to interpolate
    image_paths = [
        '/media/ws-ml/linux-drive/linux_projects/ml_projects/images/UnClipInterp/img1.png',
        '/media/ws-ml/linux-drive/linux_projects/ml_projects/images/UnClipInterp/img2.jpg'
    ]

    # Load images
    images = [Image.open(path) for path in image_paths]

    # Set the seed value for the generator
    seed = 68132

    generator = torch.Generator(device=device).manual_seed(seed)

    # Use the interpolation_factor to control interpolation strength
    # and use a loop to perform multiple interpolation steps.
    for i in range(steps):  # Adjust the loop range to run the interpolation process
        output = pipe(image=images, generator=generator)

        # Save the interpolated image
        image = output.images[0]  # The first image represents the original image (index 0)
        image.save(f"output/output_{interpolation_factor}_interp_step_{i}_image_0.jpg")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Interpolation strength and steps for image interpolation.")
    parser.add_argument("--interpolation_factor", type=float, default=0.5, help="Interpolation strength controlled by interpolation_factor.")
    parser.add_argument("--steps", type=int, default=8, help="Number of interpolation steps.")
    args = parser.parse_args()

    # Call the main function with the provided interpolation_factor and steps
    main(args.interpolation_factor, args.steps)
