# RealFill (WIP)

[RealFill](https://arxiv.org/abs/2309.16668) is a method to personalize text2image inpainting models like stable diffusion inpainting given just a few(3~5) images of a scene.
The `train_realfill.py` script shows how to implement the training procedure and adapt it for stable diffusion.


## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 

### Toy example

Now let's fill the real. For this example we will use some images of Brandenburg gate.

We already provide some images for testing in data folder

You only have to launch the training using:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="data/train"
export OUTPUT_DIR="brandenburg-model"
export VALIDATION_IMAGES="data/val/05.png"
export VALIDATION_MASKS="data/val/mask_05.png"

accelerate launch train_realfill.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --validation_images=$VALIDATION_IMAGES \
  --validation_masks=$VALIDATION_MASKS \
```

### Training on a low-memory GPU:

It is possible to run realfill on a low-memory GPU by using the following optimizations:
- [gradient checkpointing and the 8-bit optimizer](#training-with-gradient-checkpointing-and-8-bit-optimizers)
- [xformers](#training-with-xformers)
- [setting grads to none](#set-grads-to-none)

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="data/train"
export OUTPUT_DIR="brandenburg-model"
export VALIDATION_IMAGES="data/val/05.png"
export VALIDATION_MASKS="data/val/mask_05.png"

accelerate launch train_realfill.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --validation_images=$VALIDATION_IMAGES \
  --validation_masks=$VALIDATION_MASKS \
```

### Training with gradient checkpointing and 8-bit optimizers:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train realfill on a 16GB GPU.

To install `bitsandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script.

### Set grads to none

To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

## Acknowledge
This repo is built upon the code of DreamBooth from diffusers and we thank the developers for their great works and efforts to release source code. Furthermore, a special "thank you" to RealFill's authors for publishing such an amazing work.
