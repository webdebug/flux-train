import os
import sys
import argparse
import logging
import torch
from collections import OrderedDict
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import oyaml as yaml
from toolkit.job import run_job
from typing import Union

# Token setup
if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
    if not os.path.isfile("token"):
        with open("token", "w") as token_file:
            pass
        print("Token file created. Please add your Hugging Face token to the 'token' file and run the script again.")
        sys.exit(1)
    elif os.path.getsize("token") == 0:
        print("Token file is empty. Please add your Hugging Face token to the 'token' file and run the script again.")
        sys.exit(1)
    with open("token", "r") as token_file:
        os.environ["HF_TOKEN"] = token_file.read().strip()
else:
    print("Using HF_TOKEN from environment variable.")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def determine_batch_size():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = 8 * 1024**3  # 8 GB reserved
    available_memory = total_memory - reserved_memory
    batch_size = available_memory // (1024 * 1024 * 10)  # ~10 MB per 1024x1024 image
    return max(1, min(64, batch_size))

# Initialisierung
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def caption_image(image_path, model, processor, device="cuda:0", prompt="<DETAILED_CAPTION>"):
    """
    Generate a caption for a given image with optimized speed.

    Args:
        image_path (str): Path to the image.
        model: Pre-loaded model instance.
        processor: Pre-loaded processor instance.
        device (str): Device to use ("cuda:0" or "cpu").
        prompt (str): Prompt to guide the caption generation.

    Returns:
        str: Generated caption.
    """
    # Lade und konvertiere das Bild
    image = Image.open(image_path).convert('RGB')
    image_size = image.size

    # Verarbeitung der Eingaben
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device, dtype=torch.float16)
    }

    # Generiere die Caption
    with torch.amp.autocast("cuda"):  # Korrekte Nutzung der neuen API
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,  # Begrenzte Token-Länge für schnellere Generierung
            num_beams=1,         # Einfachere Hypothese für Geschwindigkeit
            do_sample=True       # Sampling anstelle deterministischer Suche
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Post-Processing der Caption
    caption_dict = processor.post_process_generation(generated_text, task=prompt, image_size=image_size)

    if isinstance(caption_dict, dict) and prompt in caption_dict:
        caption = caption_dict[prompt]
        if isinstance(caption, list):
            caption = ' '.join(caption)
    else:
        caption = str(caption_dict)

    return caption

def process_images(input_folder):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0))
    ])

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(input_folder, filename)
            caption_filename = f"{os.path.splitext(filename)[0]}.txt"
            caption_path = os.path.join(input_folder, caption_filename)

            # Überspringe, wenn die Caption bereits existiert
            if os.path.exists(caption_path):
                logging.info(f"Caption already exists for {filename}, skipping...")
                continue

            # Öffne das Bild und wende Augmentierungen an
            image = Image.open(image_path).convert('RGB')
            image = augmentations(image)

            # Rufe caption_image mit den erforderlichen Argumenten auf
            caption = caption_image(image_path, model, processor, device=device)

            # Entferne irrelevante Teile der Caption
            caption = caption.replace("of the image", "")

            if caption:
                with open(caption_path, "w", encoding='utf-8') as caption_file:
                    caption_file.write(caption)
                logging.info(f"Caption saved for {filename}")
            else:
                logging.error(f"Failed to get caption for {filename}")

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

    config_file = '/workspace/config.yaml'
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    config = load_config(config_file)

    input_folder = config.get('data_folder', '/workspace/data')
    output_folder = '/workspace/output'

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    num_images = count_images(input_folder)
    if num_images == 0:
        print(f"Error: No images found in the input folder '{input_folder}'.")
        sys.exit(1)

    trigger = args.trigger or config.get('trigger')
    prompts = config.get('prompts', [])
    if not prompts:
        print("Error: No prompts found in the config file.")
        sys.exit(1)

    process_images(input_folder)
    logging.info("Pre-process completed")

    steps = config.get('steps', num_images * 100)
    batch_size = determine_batch_size()

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
                        ('batch_size', batch_size),
                        ('steps', steps),
                        ('gradient_accumulation_steps', 4),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('gradient_checkpointing', True),
                        ('optimizer', 'adamw8bit'),
                        ('lr', config.get('lr', 4e-4)),
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
                        ('name_or_path', config.get('base_model', 'black-forest-labs/FLUX.1-dev')),
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

    run_job(job_to_run)

if __name__ == "__main__":
    main()
