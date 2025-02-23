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

warnings.filterwarnings("ignore")

# Proper if/else for device assignment
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def test(config, ds, mode, model_path):
    # Example: run multiple epochs (though it's unclear if you really want 3 test loops)

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
    output = open(f"bertpe_output/{ds}/{model_path.split('/')[-1]}", "w")

    for batch in valDataloader:
        inputs, query, MAPScore = batch
        bsz, gsz, _ = inputs["input_ids"].size()

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
    (pearsonr, pearsonp,kendalltauCorrelation, kendalltauPvalue,spearmanrCorrelation, spearmanrPvalue) = computeMetric(QPP, MAP)
    print(ds)
    print("  pearsonrCorrelation: {0:.3f}".format(pearsonr))
    print("  kendalltauCorrelation: {0:.3f}".format(kendalltauCorrelation))


if __name__ == "__main__":

    mode = "dualpairs"  # e.g., among ['dualpairs', 'double', 'single']
    mode='single'
    ds = "triviaqa"     # e.g., among ['triviaqa']

    model_path = "model/hotpotqa/adv_model_dualpairs_hotpotqa_eps_0.001_weight_0.1.model"
    model_path = "model/hotpotqa/adv_model_single_hotpotqa_eps_0.02_weight_1.0.model"
    model_path ="model/triviaqa/adv_model_single_triviaqa_eps_0.05_weight_0.5.model"
    # Ensure you read in a valid JSON file and parse it into a dict
    with open(f"config_{ds}_{mode}_optimized.json", "r") as jsonfile:
        config = json.load(jsonfile)

    # Now 'config' is a dictionary with keys like "bertModel", "seed_val", etc.
    test(config, ds, mode, model_path)
