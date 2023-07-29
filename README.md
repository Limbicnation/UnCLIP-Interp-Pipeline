# UnCLIP-Interp-Pipeline

This repository contains a diffusion pipeline for interpolating image embeddings using polynomial interpolation between two images or an `image_embeddings` tensor. The pipeline leverages the `UnCLIPImageVariationPipeline` from the Hugging Faces Diffusers library to achieve image interpolation with control over interpolation strength (`--interpolation_factor`) and the number of interpolation steps (`--steps`).
