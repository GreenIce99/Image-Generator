import torch
from diffusers import StableDiffusionPipeline

# Load the realistic model from your Google Drive path later
pipe = StableDiffusionPipeline.from_single_file(
    "/content/drive/MyDrive/AI_Models/realistic_vision_v6.0.safetensors", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# UNRESTRICTED: Disable safety checker
pipe.safety_checker = None
pipe.requires_safety_checker = False

# Recommended Realistic Vision prompt template
prompt = "RAW photo, a person standing in a park, 8k uhd, dslr, soft lighting, high quality, film grain"
image = pipe(prompt).images
image.save("output.png")
