import torch
from diffusers import StableDiffusionPipeline

# Load model (using a standard one as a base)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# THE UNRESTRICTED STEP: Disable the built-in safety filters
pipe.safety_checker = None
pipe.requires_safety_checker = False

prompt = "Your unrestricted prompt here"
image = pipe(prompt).images[0]
image.save("output.png")
