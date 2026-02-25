from my_utils import *
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import openai
from openai import OpenAI
import os
import base64
import time
import math
from io import BytesIO


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(img):
    # 创建一个内存字节流
    buffered = BytesIO()
    # 使用图像的原始格式保存
    format = img.format if img.format else "JPEG"  # 如果无法获取格式，默认使用 PNG
    img.save(buffered, format=format)
    # 获取字节数据
    img_bytes = buffered.getvalue()
    # 对字节数据进行 Base64 编码
    return base64.b64encode(img_bytes).decode('utf-8')
def resize_image(image_path, max_area=263000):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size
    if img.mode == "RGBA":
        img = img.convert("RGB")  # 转换为 RGB 模式
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
def get_response(messages):
    client = OpenAI(
        api_key="sk-cbc33252e4054c51bddbf2faccded062",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=messages,
        temperature=0,
        top_p=1,
        seed=42,
        max_tokens=100,
        n=1,
        )
    # print(completion.model_dump_json())
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="use qwen run baseline,input:image,text,question,caption")
    parser.add_argument('--test_path',type = str,help = 'input data',default = "/root/data1/Code/Multimodalqa/TextImagelistQ.json")
    args = parser.parse_args()

    set_seed(42)

    ques_data = read_json("/root/data1/Code/Multimodalqa/flod_save_models/retrieve_data/vbge_retrieve_allfold.json")

    test_data =  read_json(args.test_path)

    id2candidata = dict()
    for item in ques_data:
        data_id =item["data_id"]
        id2candidata[data_id] = item["retrieved"]

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
    
    data_inspection_failed=[]
    for idx,item in enumerate(tqdm(test_data[:])):
        while True:
            pic_path = "/root/data1/Code/Multimodalqa/MMQA/final_dataset_images/"+id2imagepath[item["supporting_context"][-1]["doc_id"]]
            ques = item["question"]
            txt_list = id2candidata[item["qid"]][:3]
            # txt_list = []
            # for txt_item in item["metadata"]["text_doc_ids"]:
            #     txt_list.append(id2text[txt_item])
            # txt = []
            # txt_id = set()
            # for txt_item in item["supporting_context"]:
            #     if txt_item["doc_part"] == "text":
            #         txt_id.add(txt_item["doc_id"])
            # for t_item in txt_id:
            #     txt.append(id2text[t_item])
            txt = "\n".join(txt_list)
            resized_img = resize_image(pic_path)
            image = resized_img
            base64_image = encode_image_from_pil(image)
            print(f"data_id: {item["qid"]} Question: {ques}\n Passage: {txt}\n")
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
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "\n"},
                    {"type": "text", "text": "Question: "},
                    {"type": "text", "text": ques},
                    {"type": "text", "text": "\n"},
                    {"type": "text", "text": "Answer: "}
                    # {"type": "text", "text": "Short Answer(Short as possible): "}
                ]
                }
            ]
            # print(messages)
            try:
                output = get_response(messages)
                data = {
                    'data_id': item["qid"], 
                    'question': ques, 
                    'prediction': output,
                    'answer': item["answers"][0]["answer"]
                }
                # print(data)
                with open('/root/data1/Code/Multimodalqa/qwenmax_result/foldtest/vbge_top3.jsonl',"a") as f:
                    f.write(json.dumps(data)+"\n")
                break
            except openai.BadRequestError as e:
                if "data_inspection_failed" in str(e):
                    data_inspection_failed.append(item["qid"])
                    print(f"Skipping question due to inappropriate content check: {data_inspection_failed}")
                else:
                    print(f"BadRequestError occurred: {e}")
                break
            except openai.RateLimitError:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(5)  # 等待 5 秒后重试当前问题