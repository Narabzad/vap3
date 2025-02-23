import numpy as np
import json
from collections import Counter
import math

def compute_query_clarity(prompt, variations):
    """
    Computes the Query Clarity Score between the main prompt and its variations.
    
    Parameters:
    - prompt: str
        The main prompt text.
    - variations: list of str
        A list of variations of the main prompt.
    
    Returns:
    - clarity_scores: list of float
        Query Clarity Scores for each variation compared to the main prompt.
    """
    def unigram_prob(text):
        words = text.split()
        total_words = len(words)
        word_counts = Counter(words)
        return {word: count / total_words for word, count in word_counts.items()}
    
    prompt_lm = unigram_prob(prompt)
    clarity_scores = []
    epsilon = 1e-10
    
    for variation in variations:
        variation_lm = unigram_prob(variation)
        score = 0.0
        for word, prob_w_prompt in prompt_lm.items():
            prob_w_variation = variation_lm.get(word, epsilon)
            score += prob_w_prompt * math.log((prob_w_prompt + epsilon) / (prob_w_variation + epsilon))
        clarity_scores.append(score)
    
    return clarity_scores


for ds in ['hotpotqa','triviaqa']:
    # Load the dataset
    with open(f'{ds}_llama3_8B_dataset_test.json', 'r') as file:
        dataset = json.load(file)

        
    output_mean=open(f'qcs_{ds}_mean.txt','w')


    for pid in dataset:
        main_prompt_id = dataset[pid]['main']['prompt_id']
        main_prompt = dataset[pid]['main']['prompt']
        variations = [v['prompt'] for v in dataset[pid]['variation']]
        variations_id = [v['prompt_id'] for v in dataset[pid]['variation']]
        all_prompts=[main_prompt] + variations
        all_prompts_id=[main_prompt_id] + variations_id
        exact_match_main = dataset[pid]['main']['exact_match']
        exact_match_var = [v['exact_match'] for v in dataset[pid]['variation']]
        exact_match_all = [exact_match_main] + exact_match_var
        query_clarity_scores=[]
        for i in range(len(all_prompts)):
            scores = compute_query_clarity(all_prompts[i],all_prompts)
            scores = [score for score in scores if score != 0]
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            std_score = np.std(scores)

            # Write outputs
            exact_match = dataset[pid]['main']['exact_match']
            output_mean.write(f"{all_prompts_id[i]} {exact_match_all[i]} {mean_score}\n")

    output_mean.close() 
