# Grounded VLM

This repository is based on [GLaMM](https://github.com/mbzuai-oryx/groundingLMM). We have added processes for training and evaluation to predict masks for target objects and placement areas.

## Setup

- Follow GLaMM's repository to prepare the [environment](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/install.md) and [data](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/datasets.md).

- Download the pre-trained model `GLaMM-GranD-Pretrained` provided by the their [repository](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/model_zoo.md).

## Training & Evaluation

- We finetune the pre-trained model for Grounded Conversation Generation using both their training data and our generated data. Run the training script: `bash run_train.sh`

- Then we post-process the saved checkpoints and run the evaluation using: `bash run_eval.sh`