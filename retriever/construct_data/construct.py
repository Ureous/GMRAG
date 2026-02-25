from my_utils import *
import argparse
from FlagEmbedding import BGEM3FlagModel
from visual_bge.modeling import Visualized_BGE
import torch
import numpy as np
from tqdm import tqdm
import json

from transformers import AutoTokenizer
def truncate_text(tokenizer,text, max_length=512):
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)
def simpos(query: str,img_path:str, pos:list, tokenizer,embv_model,num_context:int) -> list:
    pos_emb = []
    if len(pos) == 1:
        return pos
    embv_model.eval()
    with torch.no_grad():
        
        for item_sentence in pos:
            item_sentence = truncate_text(tokenizer,item_sentence)
            embedding_sentence = embv_model.encode(text= item_sentence)
            pos_emb.append(embedding_sentence)
            
        query_embedding = embv_model.encode(
            image = img_path,text = query
        )
        
    pos_emb = torch.cat(pos_emb,dim=0)
    similarity = query_embedding @ pos_emb.T
    similarity = similarity.detach().cpu().numpy()
    similarity = similarity.squeeze()
    similarpos_indices = np.argsort(-similarity)[:num_context]
    similarpos = [pos[i] for i in similarpos_indices]
    return similarpos

def simneg(pos: str, neg:list, emb_model,num_context:int) -> list:
    with torch.no_grad():
        neg_emb = emb_model.encode(
            neg
        )['dense_vecs']
        
        pos_emb = emb_model.encode(
            pos
        )['dense_vecs']
        
    similarity = pos_emb @ neg_emb.T
    similarity = similarity.squeeze()
    similarneg_indices = np.argsort(-similarity)[:num_context]
    similarneg = [neg[i] for i in similarneg_indices]
    return similarneg
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct data for retrieval")
    parser.add_argument("--data_path", type=str, required=True, help="The path of the original data")
    parser.add_argument("--output_path", type=str, required=True, help="The path of the output data")
    parser.add_argument("--emb_model", type=str, required=True, help="The path of the text embedding model")
    parser.add_argument("--embv_model", type=str, required=True, help="The path of the image and text embedding model")
    parser.add_argument("--image_path", type=str, required=True, help="The path of the train image")
    args = parser.parse_args()

    filt = read_json(args.data_path)
    emb_model = sentence_model = BGEM3FlagModel(args.emb_model,use_fp16=True)
    embv_model = Visualized_BGE(model_name_bge = "BAAI/bge-base-en-v1.5", model_weight=args.embv_model)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    for idx,item in enumerate(tqdm(filt[:])):
    """
        split context to chunks
    """
    context = item["context"]
    ans = item["answer"]
    query = item["question"]
    context_chunks = context.split("\n")
    chunks = [line for line in context_chunks if line !=""]
    pos_chunks =[]
    neg_chunks =[]
    for i in range(len(chunks)):
        if ans in chunks[i]:
            pos_chunks.append(chunks[i])
        else:
            neg_chunks.append(chunks[i])
    extend = item["image"].split('.')[-1]
    img_path = args.image_path +item["voa_image_id"]+ '.'+extend
    pos = simpos(query,img_path, pos_chunks,tokenizer, embv_model,num_context = 1)
    neg = simneg(pos, neg_chunks, emb_model,num_context=9)
    new_data = {
        'voa_example_id': item["voa_example_id"], 
        'question': query,
        'positive': pos,
        'negtive': neg,
        'caption': item["caption"],
        'voa_image_id':item["voa_image_id"],
        'answer': item["answer"],
        'context': item["context"],
    }
    append_to_json_file(new_data,args.output_path)