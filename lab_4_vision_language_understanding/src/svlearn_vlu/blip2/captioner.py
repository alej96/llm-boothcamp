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
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig
import json
from svlearn_vlu import config
from tqdm import tqdm
from svlearn.common.utils import file_exists

#  -------------------------------------------------------------------------------------------------

class Blip2Captioner:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        """
        Initializes the BlipCaptioner with a pre-trained BLIP model and its processor.
        
        Args:
            model_name (str): The name of the BLIP model to load. Default is 
            "Salesforce/blip2-opt-2.7b".
        """
        # Determine device and configure accordingly
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            quantization_config = None  # MPS doesn't support 8-bit quantization
            torch_dtype = torch.float32  # MPS works better with float32
        else:
            self.device = torch.device("cpu")
            quantization_config = None
            torch_dtype = torch.float32

        # Load the processor for BLIP-2
        self.processor = Blip2Processor.from_pretrained(model_name)

        # Configure the model based on device
        model_kwargs = {
            "torch_dtype": torch_dtype
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            **model_kwargs
        )
        
        # Move model to device (skip if using 8-bit quantization, as model is already on correct device)
        if quantization_config is None:
            self.model.to(self.device)

    #  -------------------------------------------------------------------------------------------------

    def _generate_caption(self, image, prompt: str = "A photo of "):

        # Preprocess the image and generate captions
        inputs = self.processor(image, prompt, return_tensors="pt")
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        out = self.model.generate(**inputs)
        
        # Decode the generated caption
        return self.processor.decode(out[0], skip_special_tokens=True)
    

    def generate_captions(self, image_dir, save_json=False, output_file="captions.json"):
        """
        Generates captions for all images in the specified directory and optionally saves them as a JSONL file.
        
        Args:
            image_dir (str): Path to the directory containing images.
            save_jsonl (bool): If True, saves the captions as a JSON file. Default is False.
            output_file (str): Name of the output JSON file. Default is "captions.json".
        
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

                image_caption['blip2_caption'] = caption
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
    captioner = Blip2Captioner()
    captions = captioner.generate_captions(data_dir, 
                                           save_json=True, 
                                           output_file=f"{data_dir}/captions.json")
    
    for filename , caption in captions.items():
        print(f"Image: {filename}, Caption: {caption['blip2_caption']}")

#  -------------------------------------------------------------------------------------------------