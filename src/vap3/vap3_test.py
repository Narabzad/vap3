import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from dataset import Dataset
import torch
from utils import computeMetric
import random
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

# Proper if/else for device assignment
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def test(config, ds, mode, model_path):
    # Load tokenizer from config
    tokenizer = BertTokenizer.from_pretrained(config["bertModel"])
    # Load model
    model = torch.load(model_path)
    model.to(device)
    
    # Set random seeds
    random.seed(config["seed_val"])
    np.random.seed(config["seed_val"])
    torch.manual_seed(config["seed_val"])
    torch.cuda.manual_seed_all(config["seed_val"])
    
    # Create validation/test dataset
    datasetVal = Dataset(
        dataPath=config["dataPath"],
        tokenizer=tokenizer,
        type="test",
        ds=ds,
        mode=mode
    )
    print("{:>5,} validation samples".format(len(datasetVal)))
    
    # Create DataLoader
    valDataloader = DataLoader(
        dataset=datasetVal,
        batch_size=config["batch"],
        shuffle=True,
        collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id)
    )
    
    # Dictionaries to track predicted and true values
    QPP = {}
    MAP = {}
    
    # Open output file
    output = open(f"output_model_path.split('/')[-1]}", "w")
    
    for batch in valDataloader:
        inputs, query, MAPScore = batch
        bsz, gsz, * = inputs["input_ids"].size()
        
        # Flatten group dimension
        inputs = {
            k: v.view(bsz * gsz, -1)
            for k, v in inputs.items()
        }
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids"),
        }
        
        with torch.no_grad():
            qpp_logits = model.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"]
            )
            
        # Store results per query
        for q in range(len(query)):
            MAP[query[q]] = float(MAPScore[q][0])
            QPP[query[q]] = float(qpp_logits[q].item())
            # Write output
            output.write(
                str(query[q])
                + "\t"
                + str(MAP[query[q]])
                + "\t"
                + str(qpp_logits[q].item())
                + "\n"
            )
            
    # Close file
    output.close()
    
    # Compute correlation metrics
    (pearsonr, pearsonp, kendalltauCorrelation, kendalltauPvalue, spearmanrCorrelation, spearmanrPvalue) = computeMetric(QPP, MAP)
    print(ds)
    print("  pearsonrCorrelation: {0:.3f}".format(pearsonr))
    print("  kendalltauCorrelation: {0:.3f}".format(kendalltauCorrelation))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test script with command line arguments')
    parser.add_argument('--mode', type=str, default='dualpairs',
                      help='Mode of operation (e.g., dualpairs, double, single)')
    parser.add_argument('--ds', type=str, required=True,
                      help='Dataset name (e.g., triviaqa)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model file')
    
    args = parser.parse_args()
    
    # Read config file
    with open(f"config_{args.ds}_{args.mode}_optimized.json", "r") as jsonfile:
        config = json.load(jsonfile)
    
    # Run test with provided arguments
    test(config, args.ds, args.mode, args.model_path)
