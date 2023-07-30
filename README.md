# UnCLIP-Interp-Pipeline

This repository contains a diffusion pipeline for interpolating image embeddings using polynomial interpolation between two images or an `image_embeddings` tensor. The pipeline leverages the `UnCLIPImageVariationPipeline` from the Hugging Faces Diffusers library to achieve image interpolation with control over interpolation strength (`--interpolation_factor`) and the number of interpolation steps (`--steps`).

# Define Images to Interpolate

| Input Image | Interpolated Image |
|:-------------:|:--------------:|
| ![Bear Resized Image](https://github.com/Limbicnation/UnCLIP-Interp-Pipeline/blob/main/images/bear_resized_256.png) | ![Bear Upscaled Image](https://github.com/Limbicnation/UnCLIP-Interp-Pipeline/blob/main/images/bear_output_Upscaled_256.gif) |

# UnCLIP-Interp

GitHub Repository: https://github.com/Limbicnation/UnCLIP-Interp-Pipeline.git

⚠️ Warning: The download size for this project might be over 8GB. Please ensure you have sufficient disk space before proceeding with the installation.

1. Create and activate a new conda environment:

```bash
conda create -n unclip-interp python=3.10
conda activate unclip-interp
```
Clone the 'diffusers' repository:
``
git clone https://github.com/huggingface/diffusers.git``

Install additional dependencies:
```
pip install numpy
pip install transformers diffusers
```
Update 'diffusers' to the latest version:
```
pip install --upgrade diffusers
```
Install 'diffusers' with PyTorch support:
```
pip install --upgrade diffusers[torch]
```
Run the UnCLIP interpolation script:
```
python unclip_interpolation.py --xformers 🔥

```
Run the UnCLIP interpolation script with arguments:
```
python unclip_interpolation_copy.py --interpolation_factor 0.75
python unclip_interpolation_copy.py --interpolation_factor 0.5 --steps 8
```
Run the image upscaling pipeline:
```
python image_upscaling_pipeline.py
```

Define images for interpolation in the 'output' folder.
Enjoy exploring UnCLIP-Interp!
```
Please make sure to adapt the instructions to your specific needs if required.
```
