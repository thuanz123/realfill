import argparse
import os

import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--validation_image",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation image",
)
parser.add_argument(
    "--validation_mask",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation mask",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument("--seed", type=int, default=1337, 
                    help="A seed for reproducible inference.")
parser.add_argument(
    "--num_output_images",
    type=int,
    default=5,
    help="The number of output results to generate",
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=100,
    help="The number of inference step when generating new images",
)

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    generator = None 

    # create & load model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        revision=None
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", revision=None,
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", revision=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    # image = Image.open(args.validation_image)
    # mask_image = Image.open(args.validation_mask)
    image = Image.open(args.validation_image).convert('RGB')
    mask_image = Image.open(args.validation_mask).convert('RGB')

    results = pipe(
        ["a photo of sks"] * args.num_output_images, image=image, mask_image=mask_image, 
        num_inference_steps=args.num_inference_steps, guidance_scale=1, generator=generator, 
    ).images

    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)
    
    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)

    for idx, result in enumerate(results):
        # print(image.size)
        # print(mask_image.size)
        # print(result.size)
        # result = Image.composite(result, image, mask_image)
        result.save(f"{args.output_dir}/{idx}.png")

    del pipe
    torch.cuda.empty_cache()
