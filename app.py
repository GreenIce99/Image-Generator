import torch
from diffusers import StableDiffusionPipeline

# This setup bypasses the standard filters
def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Disable the safety checker to allow all content
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    image = pipe(prompt).images[0]
    image.save("output.png")

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    generate_image(user_prompt)
