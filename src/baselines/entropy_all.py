import numpy as np
import json
from transformers import BertTokenizer, BertModel
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, set the device
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Print CUDA device info
if cuda_available:
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")

# Load pre-trained model tokenizer and model
for model_name  in ['bert-base-uncased']:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    model_name = model_name.split('/')[-1]

    def generate_embeddings(texts):
        # Encode the list of texts in a batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}  

        # Get embeddings from the model
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**inputs)
        
        # Use the mean of the last hidden state outputs as embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # Move back to CPU for NumPy
        return embeddings



    def compute_entropy(embeddings):
       norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
       similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
       prob_matrix = np.exp(similarity_matrix) / np.sum(np.exp(similarity_matrix), axis=1, keepdims=True)
       entropy = -np.sum(prob_matrix * np.log(prob_matrix + 1e-10)) / len(embeddings)
       return entropy



    for ds in ['hotpotqa','triviaqa']:
        with open(f'{ds}_llama3_8B_dataset_test.json', 'r') as file:
            dataset = json.load(file)

        output_all=open(f'entropy_all_{model_name}_{ds}_test.txt','w')
        c=0
        for pid in dataset:
            print('entropy',ds,model_name, c/len(dataset))
            c+=1
            main_prompt_id = dataset[pid]['main']['prompt_id']
            main_prompt = dataset[pid]['main']['prompt']
            variations = [v['prompt'] for v in dataset[pid]['variation']]
            variations_id = [v['prompt_id'] for v in dataset[pid]['variation']]
            all_prompts=[main_prompt] + variations
            all_prompts_id=[main_prompt_id] + variations_id
            exact_match_main = dataset[pid]['main']['exact_match']
            exact_match_var = [v['exact_match'] for v in dataset[pid]['variation']]
            exact_match_all = [exact_match_main] + exact_match_var


            # Iterate over all prompts, treating each one as the main prompt
            for i, new_main_prompt in enumerate(all_prompts):
                # Set the current prompt as the main prompt
                new_main_prompt_id = all_prompts_id[i]
                
                # The remaining prompts become variations
                new_variations = all_prompts[:i] + all_prompts[i+1:]

                # Generate embeddings
                mp_emb = generate_embeddings([new_main_prompt])
                v_emb = generate_embeddings(new_variations)
                all_embeddings = np.vstack([mp_emb, v_emb])
                variation_entropy = compute_entropy(v_emb)
                combined_entropy = compute_entropy(all_embeddings)
                output_all.write(f"{new_main_prompt_id} {exact_match_all[i]} {combined_entropy}\n")

        output_all.close()