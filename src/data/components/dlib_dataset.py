from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
from xml.etree import ElementTree as ET
import os

class DlibDataset(Dataset):
    def __init__(self, data_dir, xml_file) -> None:
        self.data_dir: str = data_dir
        self.samples: list[dict] = self.load_data(xml_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        # get sample at index
        sample: dict = self.samples[index]

        # set up image's path
        filename = sample["filename"]
        image_path = os.path.join(self.data_dir, filename)

        # open image
        image = Image.open(image_path).convert("RGB")

        # get box postition
        box_left = sample["box_left"]
        box_top = sample["box_top"]
        box_width = sample["box_width"]
        box_height = sample["box_height"]

        # crop image
        cropped_image = image.crop( (box_left, box_top, box_left+box_width, box_top+box_height) )

        return cropped_image, sample["landmarks"]
    
    @staticmethod
    def annotate_image(cropped_image, landmarks: np.array):
        """Draw landmarks on image"""
        # create an ImageDraw object
        draw = ImageDraw.Draw(cropped_image)

        # draw landmarks on image
        for x, y in landmarks:
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 255, 0))

        return cropped_image

    def load_data(self, xml_file) -> list[dict]:
        # set up path to xml file
        xml_path = os.path.join(self.data_dir, xml_file)

        # set ET ~ xml file
        et = ET.parse(xml_path)

        # find root in xml file
        root = et.getroot() # <dataset>

        # find <images> in xml file
        images = root.find("images")
        
        # set up samples
        samples: list[dict] = [self.cropped_labeled_sample(image) for image in images]

        return samples

    def cropped_labeled_sample(self, image: ET.ElementTree) -> dict:
        # find <box> in xml file
        box = image.find("box")

        # set up landmarks
        landmarks = np.array( [ [float(part.attrib["x"]), float(part.attrib["y"])] for part in box ] )
        
        # crop (landmarks: pixels cordinate)
        box_top = int( box.attrib["top"] )
        box_left = int( box.attrib["left"] )
        landmarks -= np.array( [box_left, box_top] )

        return dict(
            filename = image.attrib["file"],
            width = int( image.attrib["width"] ),
            height = int( image.attrib["height"] ),
            box_top = box_top,
            box_left = box_left,
            box_width = int( box.attrib["width"] ),
            box_height = int( box.attrib["height"] ),
            landmarks = landmarks # np.array
        )