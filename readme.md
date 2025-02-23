#VAP3: Variation-Aware Prompt Performance Prediction
This repository contains **VAP3**, a **variation-aware prompt performance prediction** method that leverages prompt variations and employs **adversarial training**.

## **Training**  
To perform hyperparameter search, run:  
```python vap3_train_adversarial_hyper_search.py```

If you prefer not to train the model and perform hyperparameter search, you can download the pre-trained models:
[trained on triviaqa]()
[traind on hotpotqa]()

## Inference
```python vap3_test.py --dataset [triviaqa or hotpotqa] --model_path [model address]```

## baselines
The results of baseline models are available in the ```baseline``` directory. The corresponding code can be found in ```src/baseline```.


## data

## Evaluation
