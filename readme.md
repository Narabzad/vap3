# VAP3: Variation-Aware Prompt Performance Prediction

This repository contains **VAP3**, a **variation-aware prompt performance prediction** method that leverages prompt variations and employs **adversarial training**.

## 📂 Data
The dataset is borrowed from [PromptSET](https://github.com/Narabzad/prompt-sensitivity) and reformatted for this project.

The `data` directory includes training and test sets for HotpotQA and TriviaQA:

- `vap3_hotpotqa_llama3_8B_dataset_train.json`
- `vap3_hotpotqa_llama3_8B_dataset_test.json`
- `vap3_triviaqa_llama3_8B_dataset_train.json`
- `vap3_triviaqa_llama3_8B_dataset_test.json`

## 🚀 Training  
To perform a **hyperparameter search**, run:  
```python vap3_train_adversarial_hyper_search.py```

If you prefer not to train the model and perform hyperparameter search, you can download the pre-trained models:

[trained model on triviaqa]()

[traind model on hotpotqa]()

## Inference
```python vap3_test.py --dataset [triviaqa or hotpotqa] --model_path [model address]```

## baselines
The results of baseline models are available in the ```baseline``` directory. The corresponding code can be found in ```src/baseline```.


## Evaluation
