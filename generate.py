from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt, output_path="output.png"):
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    generate_image("A futuristic city at night")
