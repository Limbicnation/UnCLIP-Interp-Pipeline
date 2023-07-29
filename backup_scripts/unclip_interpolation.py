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
# define image to interpolate
images = [Image.open('/art.jpg'), Image.open('/bear6.png')]

# Set the seed value for the generator
seed = 68132
interpolation_methods = ['nearest', 'lanczos', 'bicubic']

generator = torch.Generator(device=device).manual_seed(seed)

for steps in [16]:
    output = pipe(image=images, steps=steps, generator=generator)

    # create a folder to store the output images
    if not os.path.exists('../output'):
        os.makedirs('../output')
    # save images indirectory
    for i, image in enumerate(output.images):
        image.save(f"/media/ws-ml/linux-drive/linux_projects/ml_projects/Output/UnCLIP-Interep/output_10_{steps}_steps_{i}.jpg")
