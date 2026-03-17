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
        #train mumuqa train data
        print("**************************")
        print(jsonl_dir)
        # data1/Code/Mumu/processdata/Mumu_rebuttal_traindata.jsonl
        # raw_pos_neg_img.json
        # cirr_data_path = os.path.join(jsonl_dir[0], "pos_neg_517.json")
        cirr_data_path = os.path.join(jsonl_dir[0], "posneg_inans_mumu(bge-m3).json")

        print(cirr_data_path)
        self.hn_mining = False # True if use "cirr/query_train_hn_mining.jsonl"
        
        self.cirr_dataset = datasets.load_dataset('json', data_files=cirr_data_path, split='train')    
        
        self.total_len = len(self.cirr_dataset)
          
        self.image_processor = image_processor
        
    def img2pil(self, image_path):
        complelte_img_path = os.path.join(self.image_dir, image_path)
        return Image.open(complelte_img_path)
    
    def __getitem__(self, item):
        q_img = self.cirr_dataset[item]["voa_image_id"]
        q_text = "Caption : "+ self.cirr_dataset[item]["caption"] +"Question : " + self.cirr_dataset[item]["question"]
        q_img_path = q_img +"."+ self.cirr_dataset[item]["image"].split('.')[-1]
        q_img = self.image_processor(self.img2pil(q_img_path))
        positive_text = self.cirr_dataset[item]["positive"]
        # positive_text = random.sample(self.cirr_dataset[item]["positive"], 1)

        # positive_img = self.cirr_dataset[item]["positive_value"]
        # positive_img = [self.image_processor(self.img2pil(positive_img))]
        # if not self.hn_mining:
        #     hn_images = random.sample(self.cirr_dataset[item]["hn_image"], self.train_group_size - 1)
        # else:
        #     per_select_num = (self.train_group_size - 1) // 2
        #     hn_images_1 = random.sample(self.cirr_dataset[item]["hn_image"], per_select_num)
        #     hn_images_2 = random.sample(self.cirr_dataset[item]["hn_mining_images"][:20], per_select_num)
            
        #     hn_images = hn_images_1 + hn_images_2
        if not self.hn_mining:
            # hn_texts = random.sample(self.cirr_dataset[item]["negtive"], self.train_group_size - 1)
            # print("-----hnming==False, in upper -----")
            # per_select_num = (self.train_group_size - 1)
            # negative_list = self.cirr_dataset[item]["negtive"]
            #         # print("negative_list: ",len(negative_list))
            # if len(negative_list) >= per_select_num:
            #     # 如果负样本足够，无放回抽样（不重复）
            #     hn_texts = random.sample(negative_list, per_select_num)
            # else:
            #     # 如果负样本不足，有放回抽样（允许重复）
            #     hn_texts = random.choices(negative_list, k=per_select_num)
            hn_texts = random.sample(self.cirr_dataset[item]["negtive"], self.train_group_size - 1)
        else:
            # per_select_num = (self.train_group_size - 1) // 2
            # hn_texts_1 = random.sample(self.cirr_dataset[item]["negtive"], per_select_num)
            # hn_texts_2 = random.sample(self.cirr_dataset[item]["negtive"][:10], per_select_num)
            
            # hn_texts = hn_texts_1 + hn_texts_2

            per_select_num = (self.train_group_size - 1)
            negative_list = self.cirr_dataset[item]["negtive"]
            if len(negative_list) >= per_select_num:
                # 如果负样本足够，无放回抽样（不重复）
                hn_texts = random.sample(negative_list, per_select_num)
            else:
                # 如果负样本不足，有放回抽样（允许重复）
                hn_texts = random.choices(negative_list, k=per_select_num)

        # hn_images = [self.image_processor(self.img2pil(_hn)) for _hn in hn_images]

        # if not self.hn_mining:
        #     # hn_texts = random.sample(self.cirr_dataset[item]["negtive"], self.train_group_size - 1)

        #     # per_select_num = (self.train_group_size - 1)
        #     # negative_list = self.cirr_dataset[item]["negtive"]
        #     per_select_num = (self.train_group_size - 3)
        #     negative_list = self.cirr_dataset[item]["middle"]
        #     if len(negative_list) != 0:
        #             # print("negative_list: ",len(negative_list))
        #         if len(negative_list) >= per_select_num:
        #             # 如果负样本足够，无放回抽样（不重复）
        #             hn_texts = random.sample(negative_list, per_select_num)
        #         else:
        #             # 如果负样本不足，有放回抽样（允许重复）
        #             hn_texts = random.choices(negative_list, k=per_select_num)
        #     else:
        #         hn_texts =[]
        # else:
        #     # per_select_num = (self.train_group_size - 1) // 2
        #     # hn_texts_1 = random.sample(self.cirr_dataset[item]["negtive"], per_select_num)
        #     # hn_texts_2 = random.sample(self.cirr_dataset[item]["negtive"][:10], per_select_num)
            
        #     # hn_texts = hn_texts_1 + hn_texts_2

        #     per_select_num = (self.train_group_size - 1)
        #     negative_list = self.cirr_dataset[item]["negtive"]
        #     if len(negative_list) >= per_select_num:
        #         # 如果负样本足够，无放回抽样（不重复）
        #         hn_texts = random.sample(negative_list, per_select_num)
        #     else:
        #         # 如果负样本不足，有放回抽样（允许重复）
        #         hn_texts = random.choices(negative_list, k=per_select_num)

        # # hn_images = [self.image_processor(self.img2pil(_hn)) for _hn in hn_images]

        # if len(hn_texts) == 0:
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][0])
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][1])
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][0])
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][1])
        # else:
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][0])
        #     hn_texts.append(self.cirr_dataset[item]["negtive"][1])
        # print("len_hn: ",len(hn_texts))
        text_candidates = positive_text + hn_texts    
        return q_img, q_text, text_candidates
        
    
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