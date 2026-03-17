import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import sys
print(os.getcwd())
sys.path.append('./FlagEmbedding/visual')

import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# from FlagEmbedding import FlagModel
from flag_eva_token_new import Flag_bgev_model
# from flag_clip import Flag_clip
import json

logger = logging.getLogger(__name__)


@dataclass
class Args:
    resume_path: str = field(
        default="/root/data1/Code/Mumu/save_models/checkpoint-1900/BGE_EVA_Token.pth", 
        metadata={'help': 'The model checkpoint path.'}
    )
    image_dir: str = field(
        default="/root/data5/MuMuQA/resize_images263000/",
        metadata={'help': 'Where the images located on.'}
    )
    encoder: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=512, 
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

def search(model: Flag_bgev_model, queries: datasets, corpus: datasets.Dataset ,k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    model.eval()
    with torch.no_grad():
        query_embeddings = model.encode_queries([queries["question"],queries["voa_image_id"]], batch_size=batch_size, max_length=max_length, query_type="mm_it")
        
        query_size = len(query_embeddings)
        text_corpus = corpus
        
        text_corpus_embeddings = model.encode_corpus(text_corpus, batch_size=batch_size, max_length=max_length, corpus_type='text')
        print(text_corpus_embeddings.size())
        print(query_embedding.size())

        similarity = query_embedding @ text_corpus_embeddings.T
        # print(query_embedding.shape)
        # print(all_embeddings.shape)
        print(similarity.shape)
        print(type(similarity))
        similarity = similarity.detach().cpu().numpy()
        similarity = similarity.squeeze()
        top_indices = np.argsort(-similarity)[:self.num_context]
        # 获取对应的句子
        top_sentences = [text_corpus["text"][i] for i in top_indices]

        return top_sentences
        
    
    
def evaluate(preds, labels, cutoffs=[1,5,10,20,50,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    easy_recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list):
            label = [label]
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
            if len(recall) > 0:
                easy_recalls[k] += 1
    recalls /= len(preds)
    easy_recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall
    
    for i, cutoff in enumerate(cutoffs):
        easy_recall = easy_recalls[i]
        metrics[f"Easy_Recall@{cutoff}"] = easy_recall

    return metrics


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    eval_data = datasets.load_dataset('json', data_files="/root/data1/Code/Mumu/processdata/retrive_goundtruth_ansnoNone.json", split='train')
    # mm_it_corpus = datasets.load_dataset('json',  data_files="the_path_to/mm_it_corpus.jsonl", split='train')
    text_corpus = datasets.load_dataset('json', data_files="/root/data1/Code/Mumu/processdata/retrive_goundtruth_ansnoNone.json", split='train')
    
    model = Flag_bgev_model(model_name_bge = "BAAI/bge-base-en-v1.5",
                        model_name_eva = "EVA02-CLIP-B-16", # "EVA02-CLIP-B-16",
                        normlized = True,
                        eva_pretrained_path = "eva_clip",
                        resume_path=args.resume_path,
                        image_dir=args.image_dir,
                        )
    
    print(args.resume_path)
    
    

    top_sentences = search(
        model=model, 
        queries=eval_data, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["candidates"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("Hybrid Corpus (All):")
    print(metrics)

    text_queries = eval_data.filter(lambda sample: sample['type'] == "text")
    mm_it_queries = eval_data.filter(lambda sample: sample['type'] == "mm_it")
    
    scores, text_indices = search(
        model=model, 
        queries=text_queries, 
        faiss_index=faiss_index_text, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    retrieval_results = []
    for indice in text_indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(text_corpus[indice]["content"])

    ground_truths = []
    for sample in text_queries:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("text tasks:")
    print(metrics)


    scores, mm_it_indices = search(
        model=model, 
        queries=mm_it_queries, 
        faiss_index=faiss_index_mm_it, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    retrieval_results = []
    for indice in mm_it_indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(mm_it_corpus[indice]["content"])

    ground_truths = []
    for sample in mm_it_queries:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("mm_it tasks:")
    print(metrics)
    
if __name__ == "__main__":
    main()