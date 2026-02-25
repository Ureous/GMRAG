from my_utils import *
from models.llava import load_llava,llavaInference
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="use qwen run baseline,input:image,text,question,caption")
    parser.add_argument('--model_path', type=str, help='model dir',default= "/root/data3/llava-hf.llama3-llava-next-8b-hf")
    parser.add_argument('--test_path',type = str,help = 'input data',default = "/root/data5/MuMuQA/eval/test.json")
    args = parser.parse_args()

    set_seed(42)

    model_path = args.model_path
    model,processor = load_llava(model_path)
    model.eval()
    model.tie_weights()
    ques_data = read_json("/root/data1/Code/GMRAG/retriever/eval/train_422_mumu_step2900.json")

    test_data =  read_json(args.test_path)
    id2candidata = dict()
    for item in ques_data:
        data_id =item["data_id"]
        id2candidata[data_id] = item["retrieved"]
        
    for idx,item in enumerate(tqdm(test_data[:])):
        extend = item["image"].split('.')[-1]
        pic_path = "/root/data5/MuMuQA/resize_images263000/"+item["voa_image_id"]+ '.'+extend
        ques = item["question"]
        txt_list = id2candidata[item["id"]][:7]
        txt = ("\n").join(txt_list)
        image = Image.open(pic_path)
        cap = item["caption"]
        messages = [
            {
            "role": "system",
            "content": [
                # {"type": "text", "text": "Instruction: Please answer the following question based on the provided image, caption and passage. If the passage does not contain the answer, respond with 'None'. If the passage contains the answer, extract and output the answer directly from the passage."},
                # {"type": "text", "text": "\n"},
                {"type": "text", "text": "Caption: "},
                {"type": "text", "text": cap},
                {"type": "text", "text": "\n"},
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
                {"type": "text", "text": "Instruction: Please answer the following question based on the provided image, caption and passage. If the passage does not contain the answer, respond with 'None'. If the passage contains the answer, extract and output the answer directly from the passage."},
                {"type": "text", "text": "\n"},
                {"type": "text", "text": "Extracted Answer(short as possible): "}
            ]
            }
        ]
        output = llavaInference(messages,image,model,processor)
        data = {
            'data_id': item["id"], 
            'question': ques, 
            'prediction': output,
            'answer': item["answer"]
        }
        # print(data)
        with open('/root/data1/Code/GMRAG/testqa/MuMuQA/llava_mumu422_top7_2900.jsonl',"a") as f:
            f.write(json.dumps(data)+"\n")

