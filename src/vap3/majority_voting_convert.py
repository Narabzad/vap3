import numpy as np
import json
import os
import argparse

def process_file(input_file,ds):
    # Extract dataset name from the path
    input_file_name = input_file.split('/')[-1]  # Assumes path structure like 'bertpe_output/dataset/file'
    
    # First load actual values
    actual_values = {}
    with open(f'baselines/{ds}/random_{ds}_test.txt') as f:
        for line in f:
            parts = line.strip().split()
            prompt_id = parts[0]
            actual_value = parts[1]
            actual_values[prompt_id] = actual_value

    # Process the input file
    votes = {}
    file = os.path.basename(input_file)
    
    for line in open(input_file, 'r').readlines():
        qid = line.split()[0]
        qid = qid.replace('alt_', '')
        
        if qid.count('_') == 2:
            main_qid = qid.replace('_'+qid.split('_')[-1], '')
        elif qid.count('_') == 1:
            main_qid = qid
        
        if main_qid not in votes:
            votes[main_qid] = []
        votes[main_qid].append(float(line.split()[2]))

    
    # Write results
    output_file = f'{input_file_name}_majority_mean.txt'
    with open(output_file, 'w') as output_file_mean:
        for id in votes:
            try:
                output_file_mean.write(f'{id}\t{actual_values[id]}\t{max(votes[id])}\n')
            except KeyError:
                print(f"Warning: Missing actual value for ID: {id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, 
                        help='Path to the input file ')
    parser.add_argument('dataset', type=str, 
                        help='triviaqa or hotpotqa')
    args = parser.parse_args()
    
    # Verify file exists and contains 'adv'
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    

    process_file(args.input_file,args.dataset)