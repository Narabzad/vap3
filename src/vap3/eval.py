import os
import io
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc

def evaluate_model(dataset: str, file_path: str):
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print("AUC Accuracy  F1 Accuracy  Recall Precision")
    
    with open(file_path, 'r') as f:
        valid_lines = [line for line in f if len(line.split()) == 3]
    
    if len(valid_lines) == 0:
        print("No valid lines found")
        return
    
    df = pd.read_csv(io.StringIO("\n".join(valid_lines)), 
                     header=None, names=["ID", "Actual", "Predicted"], sep='\s+')
    df = df[~df['ID'].str.contains("alt_")]
    
    fpr, tpr, thresholds = roc_curve(df['Actual'], df['Predicted'])
    roc_auc = auc(fpr, tpr)
    avg_threshold = np.mean(df['Predicted'])
    df['Predicted_Label'] = df['Predicted'].apply(lambda x: 1 if x > avg_threshold else 0)
    
    actual = df['Actual']
    predicted = df['Predicted_Label']
    accuracy_avg_t = accuracy_score(actual, predicted)
    f1_avg_t = f1_score(actual, predicted)
    precision_avg_t = precision_score(actual, predicted)
    recall_avg_t = recall_score(actual, predicted)
    
    print(f"{file_path} {roc_auc:.3f} {accuracy_avg_t:.3f} {f1_avg_t:.3f} {recall_avg_t:.3f} {precision_avg_t:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("file_path", type=str, help="File name to evaluate")
    args = parser.parse_args()
    os.system(f'python majority_voting_convert.py --dataset {args.daatset} --file_path {args.file_path}') 
    evaluate_model(args.dataset, args.file_path+'_majority_mean.txt')
