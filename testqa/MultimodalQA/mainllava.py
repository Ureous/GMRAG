from my_utils import *
from models.llava import load_llava,llavaInference
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import math


def resize_image(image_path, max_area=263000):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算当前图片面积
    current_area = width * height
    
    # 如果面积小于等于最大值，直接返回原图
    if current_area <= max_area:
        return img
    
    # 计算缩放比例
    scale = math.sqrt(max_area / current_area)
    
    # 计算新的宽度和高度
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 进行缩放
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    # print(new_width,new_height,new_width*new_height)
    return resized_img
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="use llava run baseline,input:image,text,question,caption")
    parser.add_argument('--model_path', type=str, help='model dir',default= "/root/data3/llava-hf.llama3-llava-next-8b-hf")
    parser.add_argument('--test_path',type = str,help = 'input data',default = "/root/data1/Code/Multimodalqa/TextImagelistQ.json")
    args = parser.parse_args()

    set_seed(42)

    model_path = args.model_path
    model,processor = load_llava(model_path)
    model.eval()
    model.tie_weights()
    """
        load candidatas:
    """
    # ques_data = read_json("/root/data1/Code/Mumu/baselineretrieve/results/marvel_mmqa_gpu.json")
    ques_data = read_json("/root/data1/Code/Multimodalqa/flod_save_models/retrieve_data/modality/Tvbge_image_retrieve_allfold.json")
    id2candidata = dict()
    for item in ques_data:
        data_id =item["data_id"]
        id2candidata[data_id] = item["retrieved"]
    """
        corresponding to imgid and testid:
    """
    text_data = read_jsonl("/root/data1/Code/Multimodalqa/MMQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl")
    image_data = read_jsonl("/root/data1/Code/Multimodalqa/MMQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl")
    test_data =  read_json(args.test_path)
    id2text = dict()
    id2imagepath = dict()
    for item in text_data:
        data_id =item["id"]
        id2text[data_id] = item["text"]
    for item in image_data:
        data_id =item["id"]
        id2imagepath[data_id] = item["path"]
        
    for idx,item in enumerate(tqdm(test_data[:])):
        pic_path = "/root/data1/Code/Multimodalqa/MMQA/final_dataset_images/"+id2imagepath[item["supporting_context"][-1]["doc_id"]]
        ques = item["question"]
        txt_list = id2candidata[item["qid"]][:4]
        txt = "\n".join(txt_list)
        
        # txt_list = []
        # for txt_item in item["metadata"]["text_doc_ids"]:
        #     txt_list.append(id2text[txt_item])
        # txt = "\n".join(txt_list)

        # txt = []
        # txt_id = set()
        # txt_list = []
        # for txt_item in item["supporting_context"]:
        #     if txt_item["doc_part"] == "text":
        #         txt_id.add(txt_item["doc_id"])
        # for t_item in txt_id:
        #     txt_list.append(id2text[t_item])
        # txt = "\n".join(txt_list)
        resized_img = resize_image(pic_path)
        image = resized_img
        messages = [
            {
            "role": "system",
            "content": [
                {"type": "text", "text": "Instruction: Please answer the following question based on the provided image and passage. If the passage does not contain the answer, respond with 'None'. If the passage contains the answer, output the answer."},
                {"type": "text", "text": "\n"},
                # {"type": "text", "text": "Caption: "},
                # {"type": "text", "text": cap},
                # {"type": "text", "text": "\n"},
                {"type": "text", "text": "Passage: "},
                {"type": "text", "text": txt},
                {"type": "text", "text": "\n"},
            ]
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image: "},
                {"type": "image"},
                {"type": "text", "text": "\n"},
                {"type": "text", "text": "Question: "},
                {"type": "text", "text": ques},
                {"type": "text", "text": "\n"},
                {"type": "text", "text": "Short Answer(Short as possible): "}
            ]
            }
        ]
        # print(messages)
        output = llavaInference(messages,image,model,processor)
        data = {
            'data_id': item["qid"], 
            'question': ques, 
            'prediction': output,
            'answer': item["answers"][0]["answer"]
        }
        # print(data)
        with open('/root/data1/Code/Multimodalqa/flod_save_models/retrieve_data/modality/llava_modality/llava_GM_I_top4.jsonl',"a") as f:
            f.write(json.dumps(data)+"\n")

