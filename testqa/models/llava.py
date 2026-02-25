from my_utils import *
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import copy
import torch
import pickle
from tqdm import tqdm
import json
import os

set_seed(42)
def load_llava(model_path):
    set_seed(42)
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") 
    
    return model,processor

def llavaInference(messages,img,model,processor):
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    print(prompt)
    raw_image =img
    # Image.open(img_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=100,temperature=0.0,do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)

    return output_text

