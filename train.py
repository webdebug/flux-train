import os
import sys
import concurrent.futures
import argparse
import logging
import torch
import oyaml as yaml
from collections import OrderedDict
from PIL import Image
from toolkit.job import run_job
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Union
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Avoid fragmentation issues

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Workaround for unnecessary flash_attn requirement
def fixed_get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Load the model and processor once, so they don't need to be reloaded for every image
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16  # Use FP16 for better performance on H100
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def caption_images_batch(image_paths, prompt="<DETAILED_CAPTION>"):
    logging.info(f"Captioning batch of {len(image_paths)} images")

    # Open and process all images
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]

    try:
        inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt")
    except ValueError as e:
        logging.error(f"Error processing batch of images: {str(e)}")
        return [None] * len(image_paths)

    # Convert inputs to the correct dtype and device
    inputs = {
        "input_ids": inputs["input_ids"].to(device).long(),
        "pixel_values": inputs["pixel_values"].to(device, dtype=torch_dtype)
    }

    # Generate captions in batch
    try:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=500,  # Reduced max tokens for faster generation
            num_beams=1,  # Reduced beam search to speed up generation
            do_sample=False
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"CUDA out of memory during batch processing. Consider reducing batch size or using gradient checkpointing.")
            return [None] * len(image_paths)
        else:
            raise e

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)

    captions = []
    for i, generated_text in enumerate(generated_texts):
        caption_dict = processor.post_process_generation(generated_text, task=prompt, image_size=images[i].size)
        if isinstance(caption_dict, dict) and prompt in caption_dict:
            caption = caption_dict[prompt]
            if isinstance(caption, list):
                caption = ' '.join(caption)
        else:
            caption = str(caption_dict)
        captions.append(caption)

    return captions

def process_images_parallel(input_folder, batch_size=4):  # Reduced batch size to avoid CUDA out of memory
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # Process images in parallel batches
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on system resources
        futures = []
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [os.path.join(input_folder, f) for f in batch_files]
            futures.append(executor.submit(caption_images_batch_and_save, batch_paths))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error processing batch: {exc}")

def caption_images_batch_and_save(image_paths):
    captions = caption_images_batch(image_paths)
    for image_path, caption in zip(image_paths, captions):
        if caption:
            caption_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.txt"
            caption_path = os.path.join(os.path.dirname(image_path), caption_filename)
            if caption.startswith("The image shows a"):
                caption = "A" + caption[17:]
            caption = caption.replace("of the image", "")
            with open(caption_path, "w", encoding='utf-8') as caption_file:
                caption_file.write(caption)
            logging.info(f"Caption saved for {os.path.basename(image_path)}")
        else:
            logging.error(f"Failed to get caption for {os.path.basename(image_path)}")

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Flux.1 dev LoRA training")
    parser.add_argument("trigger", nargs='?', default=None, help="Optional trigger word")
    parser.add_argument("model_name", nargs='?', default="lora", help="Name of the model (default: lora)")
    args = parser.parse_args()

    # Static paths
    config_file = '/workspace/config.yaml'

    # Load config
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        print("Please create a 'config.yaml' file in the /workspace directory.")
        sys.exit(1)

    config = load_config(config_file)

    # Get input folder from config, with alias support
    input_folder = config.get('data_folder', config.get('input_folder', '/workspace/data'))
    output_folder = '/workspace/output'

    # Check if input folder exists and contains images
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    num_images = count_images(input_folder)
    if num_images == 0:
        print(f"Error: No images found in the input folder '{input_folder}'.")
        print("Please add some images (PNG, JPG, JPEG, or GIF) to the input folder and try again.")
        sys.exit(1)

    # Get trigger from config if not provided as argument
    trigger = args.trigger or config.get('trigger')

    # Get prompts from config
    prompts = config.get('prompts', [])
    if not prompts:
        print("Error: No prompts found in the config file.")
        print("Please add prompts to the 'config.yaml' file.")
        sys.exit(1)

    # Process images in parallel
    process_images_parallel(input_folder)
    logging.info("Pre-process completed")

    # Calculate steps (use config value if present, otherwise default)
    steps = config.get('steps', num_images * 100)

    # Prepare job configuration
    job_to_run = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', args.model_name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', output_folder),
                    ('device', 'cuda'),
                    ('trigger_word', trigger),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', 16),
                        ('linear_alpha', 16)
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),
                        ('save_every', config.get('save_every', 1000)),
                        ('max_step_saves_to_keep', config.get('max_step_saves_to_keep', 5))
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', input_folder),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', 0.05),
                            ('shuffle_tokens', False),
                            ('cache_latents_to_disk', True),
                            ('use_cached_latents', True),
                            ('resolution', [1024, 1024])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 16),
                        ('steps', steps),
                        ('gradient_accumulation_steps', 4),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('gradient_checkpointing', True),
                        ('optimizer', 'adamw8bit'),
                        ('lr', config.get('lr', 5e-4)),
                        ('lr_scheduler', 'cosine'),
                        ('lr_scheduler_params', OrderedDict([
                            ('T_max', steps),
                            ('eta_min', 1e-6)
                        ])),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),
                        ('dtype', 'bf16')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', config.get('base_model', config.get('name_or_path', 'multimodalart/FLUX.1-dev2pro-full'))),
                        ('is_flux', True),
                        ('quantize', True)
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', config.get('sample_every', 200)),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', prompts),
                        ('neg', 'low resolution, unrealistic, blurry'),
                        ('seed', 42),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', 20)
                    ]))
                ])
            ]),
            ('meta', OrderedDict([
                ('name', '[name]'),
                ('version', '1.0')
            ]))
        ]))
    ])

    # Run the job
    run_job(job_to_run)

if __name__ == "__main__":
    main()
