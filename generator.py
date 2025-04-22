from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime
import random

# Generation config
save_dir = "data/ood_fake"
os.makedirs(save_dir, exist_ok=True)

subjects = [
    "a young man", "a young woman", "an elderly person", "a teenager",
    "a person with glasses", "a person with long hair", "a person with curly hair", "a person with short hair", "a bald person"
]
expressions = [
    "neutral expression", "smiling", "laughing", "serious face", "eyes closed"
]
poses = [
    "front-facing", "3/4 angle", "looking left", "looking right", "head slightly tilted"
]
lighting = [
    "studio lighting", "natural light", "low light", "harsh shadows", "soft shadows"
]
backgrounds = [
    "plain background", "urban background", "indoor room", "outdoor park", "blurry background"
]
misc = [
    "natural skin texture", "realistic lighting", "high detail", "bokeh effect"
]

def generate_random_prompt():
    return f"portrait of {random.choice(subjects)}, {random.choice(poses)}, {random.choice(expressions)}, {random.choice(lighting)}, {random.choice(backgrounds)}, {random.choice(misc)}"

# Load model with MPS (Apple Metal)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    # no float 16 support on MPS
    torch_dtype=torch.float32,  
    safety_checker=None
)
pipe.to("mps") 

# Generate images
num_images = 100
for i in range(num_images):
    prompt = generate_random_prompt()
    print(f"[{i+1}/{num_images}] Prompt: {prompt}")
    image = pipe(prompt).images[0]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"ood_{i:04d}_{timestamp}.png"
    image.save(os.path.join(save_dir, filename))