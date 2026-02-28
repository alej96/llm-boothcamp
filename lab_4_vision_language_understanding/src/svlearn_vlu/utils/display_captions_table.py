#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import json
import random
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from IPython.display import HTML, display
from svlearn_vlu import config

class CaptionDisplay:
    def __init__(self, image_dir, json_file):
        """
        Initializes the CaptionDisplay class.

        Args:
            image_dir (str): Path to the directory containing images.
            json_file (str): Path to the JSON file containing captions.
        """
        self.image_dir = image_dir
        self.json_file = json_file
        self.captions = self._load_captions()

    def _load_captions(self):
        """
        Loads captions from the JSON file.

        Returns:
            dict: A dictionary of captions with image file names as keys.
        """
        with open(self.json_file, "r") as f:
            return json.load(f)

    def _get_random_samples(self, num_samples):
        """
        Randomly selects a subset of captions.

        Args:
            num_samples (int): Number of random samples to select.

        Returns:
            list: A list of randomly sampled caption items.
        """
        captions_list = [{"image_file": k, **v} for k, v in self.captions.items()]
        return random.sample(captions_list, min(num_samples, len(captions_list)))

    def _resize_image(self, image_path, size=(300, 300)):
        """
        Resizes an image for display.

        Args:
            image_path (str): Path to the image file.
            size (tuple): Desired size for the image. Default is (200, 200).

        Returns:
            Image: A resized PIL Image object.
        """
        image = Image.open(image_path)
        image.thumbnail(size)
        return image

    def _image_to_base64(self, image):
        """
        Converts a PIL Image to a base64-encoded string.

        Args:
            image (Image): A PIL Image object.

        Returns:
            str: A base64-encoded string representation of the image.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _prepare_table_data(self, sampled_captions, model_names=['blip']):
        """
        Prepares data for the display table.

        Args:
            sampled_captions (list): List of sampled caption items.
            model_names (list, optional): If provided, only the caption for the models in this list will be displayed.
                                       Options: model_name must be either 'blip', 'blip2'  or 'llava'. Default is ['blip'].

        Returns:
            list: A list of dictionaries containing image and caption data.
        """
        table_data = []
        for item in sampled_captions:
            image_path = os.path.join(self.image_dir, item["image_file"])
            image = self._resize_image(image_path)
            caption = ''
            for model_name in model_names:
                if model_name.lower() not in ['blip', 'blip2', 'llava', 'deepseek']:
                    raise ValueError("model_name must be either 'blip', 'blip2'  or 'llava'")
                caption += f"{model_name.upper()}: {item.get(f'{model_name.lower()}_caption', '')}"
                caption += '<br>'

            table_data.append({"Image": image, "Caption": caption})
        return table_data

    def display(self, num_samples=5, model_names=['blip'], 
                text_color='#212529', background_color='#ffffff', 
                font_size='14px', font_weight='normal', 
                cell_padding='12px', border_color='#dee2e6',
                caption_bg_color='#f8f9fa'):
        """
        Displays a table of randomly selected images and their captions in a Jupyter notebook.

        Args:
            num_samples (int): Number of random samples to display. Default is 5.
            model_names (list, optional): List of model names whose captions to display.
                                       Options: 'blip', 'blip2', 'llava', 'deepseek'. Default is ['blip'].
            text_color (str): Text color in hex format. Default is '#212529' (dark gray).
            background_color (str): Table background color in hex format. Default is '#ffffff' (white).
            font_size (str): Font size for captions. Default is '14px'.
            font_weight (str): Font weight ('normal', 'bold', etc.). Default is 'normal'.
            cell_padding (str): Padding inside table cells. Default is '12px'.
            border_color (str): Border color in hex format. Default is '#dee2e6' (light gray).
            caption_bg_color (str): Background color for caption cells. Default is '#f8f9fa' (light gray).
        """
        sampled_captions = self._get_random_samples(num_samples)
        table_data = self._prepare_table_data(sampled_captions, model_names)

        # Create a DataFrame for the table
        df = pd.DataFrame(table_data)

        # Display the table with images and captions
        def display_image_and_caption(row):
            return f'<img src="data:image/png;base64,{self._image_to_base64(row["Image"])}" width="200">', row["Caption"]

        df_display = df.apply(display_image_and_caption, axis=1, result_type="expand")
        df_display.columns = ["Image", "Caption"]

        # Generate HTML table
        html_table = df_display.to_html(escape=False, index=False)
        
        # Add custom CSS styling
        css_style = f"""
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: {background_color};
                margin: 10px 0;
            }}
            table th {{
                background-color: {caption_bg_color};
                color: {text_color};
                font-weight: bold;
                padding: {cell_padding};
                border: 1px solid {border_color};
                text-align: left;
            }}
            table td {{
                padding: {cell_padding};
                border: 1px solid {border_color};
                vertical-align: top;
            }}
            table td:last-child {{
                color: {text_color};
                font-size: {font_size};
                font-weight: {font_weight};
                background-color: {background_color};
                line-height: 1.5;
            }}
        </style>
        """
        
        # Render the styled table in the notebook
        display(HTML(css_style + html_table))

if __name__ == "__main__":
    image_dir = config["datasets"]["unsplash"]
    captions_file = f"{image_dir}/captions.json"
    caption_display = CaptionDisplay(image_dir, f"{image_dir}/captions.json")
    caption_display.display(num_samples=5, model_names=['blip'])