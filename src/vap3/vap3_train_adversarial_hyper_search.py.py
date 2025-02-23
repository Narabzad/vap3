import json
import sys
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from dataset import Dataset
from losses import QPPLoss
import torch
from model import pre_QPP
from transformers import get_linear_schedule_with_warmup
import time
from utils import format_time
import random
import numpy as np
import warnings
import os
from datetime import datetime
warnings.filterwarnings("ignore")

# Expanded hyperparameter search space
EPOCHS = [1, 2, 3, 4, 5]
EPSILON_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
ADV_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9]
LEARNING_RATES = [1e-6, 1e-5, 1e-4]
BATCH_SIZES = [8, 16, 32]
DROPOUT_RATES = [0.1, 0.2, 0.3]

# Maximum number of combinations to try
MAX_COMBINATIONS = 50  # Increased due to larger search space

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class HyperparameterTracker:
    def __init__(self, dataset, log_dir="hyperparam_logs"):
        self.log_dir = log_dir
        self.dataset = dataset
        
        # Create dataset-specific directory
        self.dataset_log_dir = os.path.join(log_dir, f"{dataset}")
        os.makedirs(self.dataset_log_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_params = None
        self.best_model_path = None
        
        # Create log file with dataset in name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.dataset_log_dir, 
            f"hyperparam_search_{dataset}_{timestamp}.log"
        )
        
        # Write initial information
        with open(self.log_file, "w") as f:
            f.write(f"Hyperparameter Search for Dataset: {dataset}\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            f.write("Search Space:\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Epsilon values: {EPSILON_VALUES}\n")
            f.write(f"Adversarial weights: {ADV_WEIGHTS}\n")
            f.write(f"Learning rates: {LEARNING_RATES}\n")
            f.write(f"Batch sizes: {BATCH_SIZES}\n")
            f.write(f"Dropout rates: {DROPOUT_RATES}\n")
            f.write(f"Max combinations: {MAX_COMBINATIONS}\n")
            f.write("="*50 + "\n\n")
        
    def log_results(self, params, std_losses, adv_losses, avg_loss, model_path):
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nStandard Losses per epoch:\n")
            for epoch, loss in enumerate(std_losses):
                f.write(f"Epoch {epoch + 1}: {loss:.4f}\n")
            
            f.write("\nAdversarial Losses per epoch:\n")
            for epoch, loss in enumerate(adv_losses):
                f.write(f"Epoch {epoch + 1}: {loss:.4f}\n")
                
            f.write(f"\nFinal Average Loss: {avg_loss:.4f}\n")
            f.write(f"Model saved at: {model_path}\n")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_params = params
                self.best_model_path = model_path
                f.write("\n*** NEW BEST MODEL! ***\n")
            
            f.write(f"\nCurrent Best Configuration:\n")
            for key, value in self.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Loss: {self.best_loss:.4f}\n")
            
    def save_final_summary(self):
        summary_file = os.path.join(
            self.dataset_log_dir, 
            f"best_params_{self.dataset}.txt"
        )
        
        with open(summary_file, "w") as f:
            f.write(f"Best Parameters for {self.dataset}\n")
            f.write(f"{'='*50}\n")
            for key, value in self.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Best Loss: {self.best_loss:.4f}\n")
            f.write(f"Best Model Path: {self.best_model_path}\n")
            f.write(f"\nSearch completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def train_with_params(config, ds, mode, params):
    """
    Training function with specific hyperparameters
    """
    # Update config with new parameters
    config['epochs'] = params['epochs']
    config['learning_rate'] = params['learning_rate']
    config['batch'] = params['batch_size']
    
    tokenizer = BertTokenizer.from_pretrained(config['bertModel'])
    model = pre_QPP(config, device)
    
    # Set dropout rate
    model.dropout.p = params['dropout_rate']
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=params['learning_rate'],
                                eps=config['epsilon_parameter'])

    datasetTrain = Dataset(dataPath=config['dataPath'],
                         tokenizer=tokenizer,
                         type='train',
                         ds=ds,
                         mode=mode)

    trainDataloader = DataLoader(dataset=datasetTrain,
                               batch_size=params['batch_size'],
                               shuffle=True,
                               collate_fn=PadCollate(tokenizer.pad_token_id,
                                                   tokenizer.pad_token_type_id))

    total_steps = len(trainDataloader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=config['num_warmup_steps'],
                                              num_training_steps=total_steps)

    std_losses_per_epoch = []
    adv_losses_per_epoch = []

    for epoch_i in range(0, params['epochs']):
        print(f"\nEpoch {epoch_i + 1} / {params['epochs']}")
        model.train()
        
        total_std_loss = 0
        total_adv_loss = 0

        for step, batch in enumerate(trainDataloader):
            if step % 40 == 0 and not step == 0:
                print(f'  Batch {step:>5,}  of  {len(trainDataloader):>5,}')

            inputs, query, MAPScore = batch
            bsz, gsz, _ = inputs["input_ids"].size()
            inputs = {k: v.view(bsz * gsz, -1) for k, v in inputs.items()}
            MAPScore = torch.from_numpy(np.array(MAPScore, dtype='float32')).to(device)

            model.zero_grad()

            # Standard forward pass
            qpp_logits = model(inputs['input_ids'].to(device),
                             inputs['attention_mask'].to(device),
                             inputs['token_type_ids'].to(device))
            qpp_loss = QPPLoss(device).loss(qpp_logits, MAPScore)
            total_std_loss += qpp_loss.item()

            # Adversarial part
            perturbed_embeddings, attn_mask, token_type = fgsm_attack(model,
                                                                     inputs,
                                                                     params['epsilon'],
                                                                     MAPScore,
                                                                     device)

            model_outputs = model.bert(inputs_embeds=perturbed_embeddings,
                                     attention_mask=attn_mask,
                                     token_type_ids=token_type)
            
            dropout = model.dropout(model_outputs.last_hidden_state[:, 0])
            adv_logits = model.regression(dropout)
            adv_loss = QPPLoss(device).loss(adv_logits, MAPScore)
            total_adv_loss += adv_loss.item()

            # Combined loss
            loss = qpp_loss + params['adv_weight'] * adv_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Record epoch losses
        avg_std_loss = total_std_loss / len(trainDataloader)
        avg_adv_loss = total_adv_loss / len(trainDataloader)
        std_losses_per_epoch.append(avg_std_loss)
        adv_losses_per_epoch.append(avg_adv_loss)
        
        print(f"Standard Loss: {avg_std_loss:.4f}, Adversarial Loss: {avg_adv_loss:.4f}")

    # Return final model path and losses
    model_path = (f"{config['outputPath']}adv_model_{mode}_{ds}_"
                 f"eps_{params['epsilon']}_weight_{params['adv_weight']}_"
                 f"lr_{params['learning_rate']}_bs_{params['batch_size']}_"
                 f"dr_{params['dropout_rate']}_ep_{params['epochs']}.model")
    torch.save(model, model_path)
    
    return model_path, std_losses_per_epoch, adv_losses_per_epoch

def hyperparam_search(config, ds, mode='dualpairs'):
    """
    Perform hyperparameter search
    """
    tracker = HyperparameterTracker(dataset=ds)
    
    # Generate parameter combinations
    param_combinations = []
    for epochs in EPOCHS:
        for epsilon in EPSILON_VALUES:
            for adv_weight in ADV_WEIGHTS:
                for lr in LEARNING_RATES:
                    for bs in BATCH_SIZES:
                        for dr in DROPOUT_RATES:
                            param_combinations.append({
                                'epochs': epochs,
                                'epsilon': epsilon,
                                'adv_weight': adv_weight,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'dropout_rate': dr
                            })
    
    random.shuffle(param_combinations)
    param_combinations = param_combinations[:MAX_COMBINATIONS]
    
    for params in param_combinations:
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"\nTrying parameters: {param_str}")
        
        try:
            model_path, std_losses, adv_losses = train_with_params(
                config, ds, mode, params
            )
            
            # Calculate final average loss
            final_avg_loss = (std_losses[-1] + params['adv_weight'] * adv_losses[-1]) / (1 + params['adv_weight'])
            
            # Log results
            tracker.log_results(params, std_losses, adv_losses, 
                              final_avg_loss, model_path)
            
        except Exception as e:
            print(f"Error with parameters {param_str}: {str(e)}")
            with open(tracker.log_file, "a") as f:
                f.write(f"\nERROR with parameters {param_str}: {str(e)}\n")
            continue
    
    # Save final summary
    tracker.save_final_summary()
    
    print("\nHyperparameter search completed!")
    print(f"Best parameters for {ds}:")
    for key, value in tracker.best_params.items():
        print(f"{key}: {value}")
    print(f"Best Loss: {tracker.best_loss:.4f}")
    print(f"Best model saved at: {tracker.best_model_path}")
    print(f"Detailed logs saved in: {tracker.log_file}")

if __name__ == '__main__':
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    for ds in ['triviaqa']:#'hotpotqa', 'triviaqa']:
        config_path = f"config_{ds}_dualpairs_optimized.json"
        print(f"\nStarting hyperparameter search for {ds}")
        
        try:
            with open(config_path, "r") as jsonfile:
                config = json.load(jsonfile)
            hyperparam_search(config, ds, 'dualpairs')
        except Exception as e:
            print(f"Error during hyperparameter search: {str(e)}")
            continue