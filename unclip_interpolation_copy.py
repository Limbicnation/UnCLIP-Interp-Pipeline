import torch
import os
from diffusers import DiffusionPipeline
from PIL import Image

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

pipe = DiffusionPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha-image-variations",
    torch_dtype=dtype,
    custom_pipeline="unclip_image_interpolation",  # use polynomial interpolation
    max_split_size_mb=256,  # set max_split_size_mb to avoid fragmentation
    interpolation_factor=0.5,  # adjust the strength of the interpolation
)

pipe.to(device)

# Define image paths to interpolate
image_paths = [
    '/media/ws-ml/linux-drive/linux_projects/ml_projects/images/UnClipInterp/art.jpg',
    '/media/ws-ml/linux-drive/linux_projects/ml_projects/images/UnClipInterp/bear6.png'
]

# Load images
images = [Image.open(path) for path in image_paths]

# Set the seed value for the generator
seed = 68132
interpolation_methods = ['nearest', 'lanczos', 'bicubic']

generator = torch.Generator(device=device).manual_seed(seed)

# set the number of steps here
for steps in [8]:
    output = pipe(image=images, steps=steps, generator=generator)

    # Create a folder to store the output images
    if not os.path.exists('output'):
        os.makedirs('output')

    # Save images in the 'output' directory
    for i, image in enumerate(output.images):
        image.save(f"output/output_10_{steps}_steps_{i}.jpg")
