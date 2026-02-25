import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from flag_eva_token_new import Flag_bgev_model
from my_utils import *
import json
def search(model: Flag_bgev_model, queryQ,queryI, corpus,k:int = 100, batch_size: int = 1, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through...
    """
    # model.eval()
    with torch.no_grad():
        query_embedding = model.encode_queries([queryQ,queryI], batch_size=batch_size, max_length=max_length, query_type="mm_it")
        text_corpus = corpus
        text_corpus_embeddings = model.encode_corpus_item(text_corpus, batch_size=batch_size, max_length=max_length, corpus_type='text')

        similarity = query_embedding @ text_corpus_embeddings.T

        similarity = similarity.squeeze()
        top_indices = np.argsort(-similarity)[:k]
        top_sentences = [text_corpus[i] for i in top_indices]

        return top_sentences
resume_path = "/root/data1/Code/Mumu/save_421_model/checkpoint-2900/BGE_EVA_Token.pth"
image_dir = "/root/data5/MuMuQA/resize_images263000/"
eval_data = datasets.load_dataset('json', data_files="/root/data1/Code/Mumu/processdata/retrive_goundtruth_ansnoNone.json", split='train')
text_corpus = datasets.load_dataset('json', data_files="/root/data1/Code/Mumu/processdata/retrive_goundtruth_ansnoNone.json", split='train')

model = Flag_bgev_model(model_name_bge = "BAAI/bge-base-en-v1.5",
                    model_name_eva = "EVA02-CLIP-B-16", # "EVA02-CLIP-B-16",
                    normlized = True,
                    eva_pretrained_path = "eva_clip",
                    resume_path=resume_path,
                    image_dir=image_dir,
                    )
for idx,item in enumerate(tqdm(list(eval_data)[:])):
    extend = item["image"].split('.')[-1]
    pic_id = item["voa_image_id"]+ '.'+extend
    ques = "Caption : "+ item["caption"] +"Question : " + item["question"]
    top_sentences = search(
            model=model, 
            queryQ=ques, 
            queryI = pic_id,
            corpus = item['text'],
            k=100,
            batch_size=1, 
            max_length=512
        )
    data = {
            'data_id': item["id"], 
            'question': ques, 
            'caption': item["caption"],
            'candidates': item["candidates"],
            'retrieved': top_sentences,
        }
    append_to_json_file(data,'./train_422_mumu_step2900.json') 