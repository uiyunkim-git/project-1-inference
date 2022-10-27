import shutil
import os
import torchvision.utils
import torch
import model
import sys
device = "cuda"
exp_dir = "fake"

generator = model.Generator().to(device)

def main() -> None:
    # Create a experiment result folder.
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Load model weights.
    state_dict = torch.load("weight.pth", map_location=device)
    generator.load_state_dict(state_dict)
    # Start the verification mode of the model.
    generator.eval()
    # Turn on half-precision inference.
    # generator.half()

    with torch.no_grad():
        for index in range(1000):
            # Create an image that conforms to the Gaussian distribution.
            fixed_noise = torch.randn([1, 300, 1, 1], device=device)
            # fixed_noise = fixed_noise.half()
            image = generator(fixed_noise)
            torchvision.utils.save_image(image, os.path.join(exp_dir, f"{index:03d}.png"))
            print(f"The {index + 1:03d} image is being created using the model...")

    os.makedirs('submission',exist_ok=True)
    os.system("zip -r submission/20229006.zip fake") 
if __name__ == "__main__":
    main()