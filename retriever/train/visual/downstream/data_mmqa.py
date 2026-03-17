import os.path
import random
from dataclasses import dataclass

import datasets
from torch.utils.data import Dataset, IterableDataset

from .arguments import DataArguments

from PIL import Image
import json
import torch
import torch.distributed

class Multimodal_Dataset(Dataset):
    def __init__(self, args:DataArguments, image_processor=None) -> None:
        # self.image_dir = os.path.join(args.train_data_image, "CIRR_images")
        self.image_dir = args.train_data_image

        self.train_group_size = args.train_group_size
        
        jsonl_dir = args.train_data 
        #train mmqa train data
        print("**************************")
        print(jsonl_dir)
        cirr_data_path = os.path.join(jsonl_dir[0], "all_data_posneg.json")

        self.hn_mining = False # True if use "cirr/query_train_hn_mining.jsonl"
        
        self.cirr_dataset = datasets.load_dataset('json', data_files=cirr_data_path, split='train')    
        
        self.total_len = len(self.cirr_dataset)
          
        self.image_processor = image_processor
        
    def img2pil(self, image_path):
        complelte_img_path = os.path.join(self.image_dir, image_path)
        return Image.open(complelte_img_path)
    
    def __getitem__(self, item):
        q_img = self.cirr_dataset[item]["image_id"]
        q_text = self.cirr_dataset[item]["question"]
        q_img_path = q_img
        q_img = self.image_processor(self.img2pil(q_img_path))
        
        positive_text = self.cirr_dataset[item]['positive']
        if not self.hn_mining:
            hn_texts = random.sample(self.cirr_dataset[item]["negtive"], self.train_group_size - 1)
        else:
            per_select_num = (self.train_group_size - 1) // 2
            hn_texts_1 = random.sample(self.cirr_dataset[item]["negtive"], per_select_num)
            hn_texts_2 = random.sample(self.cirr_dataset[item]["negtive"][:10], per_select_num)
            
            hn_texts = hn_texts_1 + hn_texts_2
        # hn_images = [self.image_processor(self.img2pil(_hn)) for _hn in hn_images]
        
        text_candidates = positive_text + hn_texts    
        return q_img, q_text, text_candidates
        
    def getorigin(self, item):
        return self.cirr_dataset[item]
    def __len__(self):
        return self.total_len


class Multimodal_Collator:
    def __init__(self, tokenizer, mmit_max_len=109, pure_text_max_len=256):
        self.tokenizer = tokenizer
        self.mmit_max_len = mmit_max_len
        self.text_max_len = pure_text_max_len
    
    def reshape_image_candidate(self, i_candidates):
        all_candidates = []
        for group in i_candidates:
            for image in group:
                all_candidates.append(image)
        return all_candidates
    
    def reshape_text_candidate(self, t_candidates):
        all_candidates = []
        for group in t_candidates:
            for text in group:
                all_candidates.append(text)
        return all_candidates
    
    def reshape_mmit_candidate(self, mm_candidates):
        all_candidates = []
        for group in mm_candidates:
            for mm in group:
                all_candidates.append(mm)
        return all_candidates
    
    
    def __call__(self, features):
        
        q_images = [f[0] for f in features]
        q_texts = [f[1] for f in features]
        text_candidates = [f[2] for f in features]
        
        
        
        q_text_collated = self.tokenizer(
            q_texts,
            padding= True, #"max_length",
            truncation=True,
            max_length=self.mmit_max_len,
            return_tensors="pt",
        )
        q_image_collated = torch.stack(q_images)
        
        
        c_texts = self.reshape_text_candidate(text_candidates)
        c_text_collated = self.tokenizer(
                c_texts,
                padding= True, #"max_length",
                truncation=True,
                max_length=self.text_max_len,
                return_tensors="pt",
            )
        # c_image_collated = torch.stack(c_images)

        return {"mm_it_query": (q_image_collated, q_text_collated), "text_candidate": c_text_collated}