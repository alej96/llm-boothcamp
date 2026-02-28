#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from svlearn.config.configuration import ConfigurationMixin
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import os
import json
from tqdm import tqdm

#  -------------------------------------------------------------------------------------------------
class DeepSeekVLCaptioner:
    def __init__(self, model_name="deepseek-ai/deepseek-vl-1.3b-base"):
        """
        Initializes the DeepSeekVLCaptioner with a pre-trained DeepSeekVL model and its processor.

        Args:
            model_name (str): The name of the DeepSeekVL model to load. Default is
            "deepseek-ai/deepseek-vl-1.3b-base".
        """

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model = vl_gpt.to(torch.bfloat16).cuda().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------

    def _generate_caption(self, image, prompt: str = "Describe the image in detail."):
        """
        Generates a caption for the given image using the DeepSeekVL model.

        Args:
            image (PIL.Image): The image to generate a caption for.
            prompt (str): The text prompt to guide the caption generation. Default is
            "Describe the image in detail."

        Returns:
            str: The generated caption (only the assistant's response).
        """
        conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{prompt}",
            "images": [image]
        },
        {
            "role": "Assistant",
            "content": ""
        }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        try:
            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True
            )

            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        except Exception as e:
            print(f"Error generating caption: {e}")
            return ""

        return answer

    # -------------------------------------------------------------------------------------------------

    def generate_captions(self, image_dir, save_json=False, output_file="captions.json"):
        """
        Generates captions for all images in the specified directory and optionally saves them as a JSON file.

        Args:
            image_dir (str): Path to the directory containing images.
            save_json (bool): If True, saves the captions as a JSON file. Default is False.
            output_file (str): Name of the output JSON file. Default is "captions.json".

        Returns:
            dict: A dictionary where keys are image file names and values are their generated captions.
        """
        # Get a list of image files in the directory
        image_files = [
            f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]

        # Load existing captions if the file exists
        if save_json and os.path.exists(output_file):
            with open(output_file, "r") as f:
                all_captions = json.load(f)
        else:
            all_captions = {}

        # Process images with a progress bar
        for image_file in tqdm(image_files, desc="Processing images", unit="image"):
            image_path = os.path.join(image_dir, image_file)

            try:
                # Open the image
                # image = Image.open(image_path).convert("RGB")
                caption = self._generate_caption(image_path)

                # Update the captions dictionary
                image_caption = all_captions.get(image_file, {})
                image_caption["deepseek_caption"] = caption
                all_captions[image_file] = image_caption

            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                exit()

        # Save captions to a JSON file if requested
        if save_json:
            with open(output_file, "w") as f:
                json.dump(all_captions, f, indent=4)
            print(f"Captions saved to {output_file}")

        return all_captions


#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()

    config_path = os.getenv('BOOTCAMP_ROOT_DIR') + '/config.yaml'
    config = ConfigurationMixin().load_config(config_path)
    data_dir = config["datasets"]["unsplash"]
    captioner = DeepSeekVLCaptioner()   
    captions = captioner.generate_captions(data_dir, save_json=True, output_file=f"{data_dir}/captions.json")

    for filename, caption in captions.items():
        print(f"Image: {filename}, Caption: {caption['deepseek_caption']}")

#  -------------------------------------------------------------------------------------------------
