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

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    parser.add_argument('--model_path', type=str, help='model dir',default= "/root/data2/Qwen.Qwen2-VL-7B-Instruct")
    parser.add_argument('--test_path',type = str,help = 'input data',default = "/root/data5/MuMuQA/eval/test.json")
    args = parser.parse_args()

    set_seed(42)

    ques_data = read_json("/root/data1/Code/Mumu/retriveAnalysis/OriginQretrieve/retriveOq_vbge.json")

    test_data =  read_json(args.test_path)
    id2candidata = dict()
    for item in ques_data:
        data_id =item["data_id"]
        id2candidata[data_id] = item["candidates"]
    data_inspection_failed=[]
    for idx,item in enumerate(tqdm(test_data[10:])):
        while True:
            extend = item["image"].split('.')[-1]
            pic_path = "/root/data5/MuMuQA/resize_images263000/"+item["voa_image_id"]+ '.'+extend
            ques = item["question"]
            txt = id2candidata[item["id"]][:7]
            # image = Image.open(pic_path)
            base64_image = encode_image(pic_path)
            # print(txt)
            txt_str = '\n'.join(txt)  # 使用空格分隔
            # txt_str = item["context"]
            # print(txt_str)
            # txt = item["context"]
            cap = item["caption"]
            print(f"Caption: {cap}\n Passage: {txt_str}\n Question: {ques}")
            messages = [
                {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Instruction: Please answer the following question based on the provided image, caption and passage. If the passage does not contain the answer, respond with 'None'. If the passage contains the answer, extract and output the answer directly from the passage."},
                    {"type": "text", "text": "\n"},
                    {"type": "text", "text": "Caption: "},
                    {"type": "text", "text": cap},
                    {"type": "text", "text": "\n"},
                    {"type": "text", "text": "Passage: "},
                    {"type": "text", "text": txt_str},
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
                ]
                }
            ]
            # print(messages)
            try:
                output = get_response(messages)
                data = {
                    'data_id': item["id"], 
                    'question': ques, 
                    'prediction': output,
                    'answer': item["answer"]
                }
                # print(data)
                with open('/root/data1/Code/Mumu/aftertraindata/aftertrainAns/qwenmax_vbgebase_top7.jsonl',"a") as f:
                    f.write(json.dumps(data)+"\n")
                break
            except openai.BadRequestError as e:
                if "data_inspection_failed" in str(e):
                    data_inspection_failed.append(item["id"])
                    print(f"Skipping question due to inappropriate content check: {data_inspection_failed}")
                else:
                    print(f"BadRequestError occurred: {e}")
                break
            except openai.RateLimitError:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(5)  # 等待 5 秒后重试当前问题