import numpy as np
from my_utils import *
import argparse
def evaluate(preds, labels, cutoffs=[1,3,5,7,9,10]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    lable0num = 0
    for pred, label in zip(preds, labels):
        if len(label) == 0:
            lable0num += 1
            continue
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
    lablenum = 0
    for pred, label in zip(preds, labels):
        if len(label) == 0:
            lablenum += 1
            continue
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
    print(lable0num,lablenum)
    return metrics
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for Recall and MRR")
    parser.add_argument("--retrieve_file", type=str, required=True, help="The file path of the retrieve results")
    parser.add_argument("--gold_file", type=str, required=True, help="The file path of the gold results")
    args = parser.parse_args()
    """
    python eval.py --retrieve_file /root/data1/Code/Multimodalqa/rebuttal/UniIR/src/uniir_clipsf.json 
    --gold_file /root/data1/Code/Multimodalqa/all_data_posneg.json
    
    python eval.py --retrieve_file /root/data1/Code/GMRAG/retriever/eval/train_422_mumu_step3150.json --gold_file /root/data1/Code/Mumu/processdata/retrive_goundtruth_ansnoNone.json
    """
    dataretrieve = read_json(args.retrieve_file)
    data1 = read_json(args.gold_file)
    ground_truths = []
    retrieval_results = []
    for sample in data1:
        ground_truths.append(sample["candidates"])
    for sample in dataretrieve:
        retrieval_results.append(sample["retrieved"])
    metrics = evaluate(retrieval_results, ground_truths)
    print("Hybrid Corpus (All):")
    print(metrics)