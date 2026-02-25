from my_utils import *
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
import copy
import torch
import pickle
from tqdm import tqdm
import json
import os

set_seed(42)
def load_Qwen(model_path):
    set_seed(42)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="float16", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model,processor

def Qwen2vlInference(messages,model,processor):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    input_ids = inputs["input_ids"]
    # token_lengths = input_ids.shape[1]
    # # print(input_ids.size())
    # print(input_ids)
    # print(f"Token长度: {token_lengths}")
    
    # inputs = inputs.to("cpu")
    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=100,
        # temperature=0.0,
        do_sample = False,
        # top_k=None,       # 确保不使用 top_k
        # top_p=None      # 确保不使用 top_p
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

