#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------
#
#  Setup for MLflow / TensorBoard reporting:
#  ----------------------------------------
#  Install optional dependencies (in your env):
#    uv add tensorboard mlflow
#
#  View logs:
#  - TensorBoard:  tensorboard --logdir <results_dir>/runs
#                  (results_dir from config; e.g. tensorboard --logdir ./results/runs)
#                  Then open http://localhost:6006
#  - MLflow:       Uses SQLite by default: <results_dir>/mlflow.db
#                  View: mlflow ui --backend-store-uri sqlite:///<results_dir>/mlflow.db
#                  Then open http://localhost:5000 (use same backend-store-uri as training).
#

import os
import time
import torch
from sentence_transformers import (SentenceTransformer, 
                                   SentenceTransformerTrainer, 
                                   SentenceTransformerTrainingArguments, 
                                   losses)

from contrastive_loss import config
from contrastive_loss.hf_text_utils import (
    get_train_test_lists, 
    tuples_list_to_dataset, 
    sampled_dataset, 
    convert_to_triplet_dataset,
    convert_to_anchor_positive_dataset,
    get_triplet_evaluator
)

if __name__ == "__main__":
    # Determine the device to use for training (MPS, CUDA, or CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else device

    # Load the pre-trained sentence transformer model
    model_name = 'BAAI/bge-base-en-v1.5'
    model = SentenceTransformer(model_name).to(device)
    
    # Retrieve training and testing data labeled with subjects
    train, test = get_train_test_lists(cfg=config)
    
    # Convert the data to Hugging Face Dataset format
    train_dataset = tuples_list_to_dataset(train)
    test_dataset = tuples_list_to_dataset(test)
    
    # Sample the datasets to a maximum of 500 samples per label
    train_dataset = sampled_dataset(train_dataset)
    test_dataset = sampled_dataset(test_dataset)

    # Create (anchor, positive) pair datasets for training and testing
    train_dataset = convert_to_anchor_positive_dataset(train_dataset)
    test_dataset = convert_to_triplet_dataset(test_dataset)

    # Define the temperature scales (temperature = 1/ scale) that we will use in the infonce loss
    scales = [5, 20]
        
    # Set training arguments based on the device
    use_cpu = False
    no_cuda = False
    if device == "cpu":
        use_cpu = True
        no_cuda = True

    # Define the directory to save training results
    results_dir = config["paths"]["results_dir"]
    results_sub_dir = config["paths"]["results_sub_dir_infonce"]

    # MLflow: use SQLite backend (recommended over filesystem) and experiment name
    mlflow_db_path = os.path.join(results_dir, "mlflow.db")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///" + os.path.abspath(mlflow_db_path))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sbert_infonce")

    training_args = SentenceTransformerTrainingArguments(
        no_cuda=no_cuda,
        use_cpu=use_cpu,
        max_steps=2000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="steps",
        weight_decay=0.01,
        metric_for_best_model="cosine_accuracy",
        greater_is_better=True,
        save_total_limit=1,  # Keep only the best model checkpoint
        report_to=["tensorboard", "mlflow"],
        logging_dir=os.path.join(results_dir, "runs"),
    )

    # Initialize the evaluator for the test dataset
    binary_acc_evaluator = get_triplet_evaluator(test_dataset=test_dataset)
    # Evaluate the model before training
    result = binary_acc_evaluator(model)
    print(f"Before training evaluation: {result}")

    for scale in scales:
        loss = losses.MultipleNegativesRankingLoss(model, scale=scale)
        results_sub_dir_scale = f"{results_sub_dir}_scale_{scale}"
        training_args.output_dir = f"{results_dir}/{results_sub_dir_scale}"
        training_args.run_name = results_sub_dir_scale
        # Set up the trainer with the model, training arguments, dataset, and loss function
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
        )

        start_time = time.time()
        # Train the model
        trainer.train()

        end_time = time.time()

        # Evaluate the model after training
        result = binary_acc_evaluator(model)
        print(f"After training for scale {scale} evaluation: {result}")
        
        # Calculate and print the total training time
        training_time = end_time - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)

        print(f"Training for scale {scale} started at: {time.ctime(start_time)}")
        print(f"Training ended for scale {scale} at: {time.ctime(end_time)}")
        print(f"Total training time for scale {scale}: {hours} hours, {minutes} minutes, {seconds} seconds")
        start_time = time.time()
        model = SentenceTransformer(model_name).to(device)
