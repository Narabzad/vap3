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
```python src/vap3/vap3_train_adversarial_hyper_search.py```

It will sweep across the following huperparameters and log the the outputs and save the best model

```EPOCHS = [1, 2, 3, 4, 5]
EPSILON_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
ADV_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9]
LEARNING_RATES = [1e-6, 1e-5, 1e-4]
BATCH_SIZES = [8, 16, 32]
DROPOUT_RATES = [0.1, 0.2, 0.3]
```


If you prefer not to train the model and perform hyperparameter search, you can download the pre-trained models:


[trained model on triviaqa](https://drive.google.com/file/d/1BzNdHKL85CV_-VxGDQiD3tcf38HRHOrJ/view?usp=sharing)

[traind model on hotpotqa](https://drive.google.com/file/d/1jUfJemJTgIjnillK9HN0ODwaomkZpcqP/view?usp=sharing)

## Inference
```python vap3_test.py --dataset [triviaqa or hotpotqa] --model_path [model address]```

## baselines
The results of baseline models are available in the ```baseline``` directory. The corresponding code can be found in ```src/baseline```.


## Evaluation
