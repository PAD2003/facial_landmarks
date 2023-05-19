import pyrootutils
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import numpy as np
from models.dlib_module import DlibModule
import sys
from models.components.simple_resnet import SimpleResnet
import os

# fix bug
sys.path.append("src/data")

# set up path
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
outputs_path = path / "deploy" / "outputs"

# create transform (same as transform_val)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
simple_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# def func to draw landmarks on image
def draw_landmarks(image: Image, input: torch.Tensor, output: torch.Tensor) -> Image:
    # remove first dimension (batch_size)
    input = input.squeeze()
    output = output.squeeze()

    # reverse color transform
    def denormalize(input: torch.Tensor, std=std, mean=mean) -> torch.Tensor:
        # clone: make a copy
        tmp = input.clone()

        # denormalize
        for t, m, s in zip(tmp, mean, std):
            t.mul_(s).add_(m)

        # clamp: limit value to [0, 1]
        return torch.clamp(tmp, 0, 1)
    input = denormalize(input)

    # get information of original input
    width, height = image.size

    # denormalized output (landmarks)
    landmarks = (output + 0.5) * np.array( [width * 224 / 256, height * 224 / 256] ) + np.array([width * 16 / 256, height * 16 / 224])

    # draw landmarks on original image
    draw = ImageDraw.Draw(image)
    for x, y in landmarks:
        draw.ellipse([(x-4, y-4), (x+4, y+4)], fill=(0, 255, 0))

    # return annotated image
    return image
    
# load model from ckpt file
# model = DlibModule.load_from_checkpoint(net=SimpleResnet(), checkpoint_path="deploy/checkpoints/last3.ckpt")
model = SimpleResnet()
model.load_state_dict(torch.load(f="deploy/checkpoints/last3.pth"))

# get all filenames
folder = "deploy/data/"
files = os.listdir(folder)
files = ["barack-obama-500.jpg"]

for file in files:
    # prepare input
    image = Image.open(folder + file).convert("RGB")
    width, height = image.size
    input = simple_transform(image)

    # use model to predict landmark
    model.eval()
    with torch.inference_mode():
        input = input.unsqueeze(dim=0)
        output = model(input)

        annotated_input_image = draw_landmarks(image, input, output).resize((width, height))

    # save image
    annotated_input_image.save(outputs_path / file)