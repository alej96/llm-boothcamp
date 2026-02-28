#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import json
from svlearn_vlu import config
from svlearn.common.utils import file_exists
from tqdm import tqdm
import torch

#  -------------------------------------------------------------------------------------------------

class BlipCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initializes the BlipCaptioner with a pre-trained BLIP model and its processor.
        
        Args:
            model_name (str): The name of the BLIP model to load. Default is 
            "Salesforce/blip-image-captioning-base".
        """
        # Load the BLIP model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else self.device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    #  -------------------------------------------------------------------------------------------------

    def _generate_caption(self, image, prompt: str = "A photo of "):

        # Preprocess the image and generate captions
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        
        # Decode the generated caption
        return self.processor.decode(out[0], skip_special_tokens=True)
    

    def generate_captions(self, image_dir, save_json=False, output_file="captions.json"):
        """
        Generates captions for all images in the specified directory and optionally saves them as a JSONL file.
        
        Args:
            image_dir (str): Path to the directory containing images.
            save_jsonl (bool): If True, saves the captions as a JSONL file. Default is False.
            output_file (str): Name of the output JSONL file. Default is "captions.jsonl".
        
        Returns:
            list: A list of dictionaries, where each dictionary contains the image file name and its generated caption.
        """
        
        # Get a list of image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if save_json and file_exists(output_file):
            with open(output_file, "r") as f:
                all_captions = json.load(fp=f)
        else: 
            all_captions = {}

        # Process images with a progress bar
        for image_file in tqdm(image_files, desc="Processing images", unit="image"):
            image_path = os.path.join(image_dir, image_file)
            
            try:
                # Open the image
                image = Image.open(image_path).convert("RGB")
                caption = self._generate_caption(image)
            
                image_caption = all_captions.get(image_file, {})

                image_caption['blip_caption'] = caption
                all_captions[image_file] = image_caption

            except Exception as e:
                print(f"Error processing image {image_file}: {e}")

        if save_json:
            with open(output_file, "w") as f:
                json.dump(all_captions , f)
        print(f"Captions saved to {output_file}")
        
        return all_captions

#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = config["datasets"]["unsplash"]
    captioner = BlipCaptioner()
    captions = captioner.generate_captions(data_dir, 
                                           save_json=True, 
                                           output_file=f"{data_dir}/captions.json")
    
    for filename , caption in captions.items():
        print(f"Image: {filename}, Caption: {caption['blip_caption']}")

#  -------------------------------------------------------------------------------------------------