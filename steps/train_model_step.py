import torch
import mlflow
from zenml import step
from datasets import DatasetDict
from config.training_config import TrainingConfig
from src.training import (
    load_training_models,
    create_compute_metrics_function,
    create_training_arguments,
    create_trainer,
    save_model_and_processor
)

@step(experiment_tracker="mlflow_tracker")  
def train_model_step(dataset: DatasetDict, config: TrainingConfig) -> str:
    
    # Create unique model name based on key parameters
    model_suffix = f"lr{config.learning_rate:.0e}_bs{config.per_device_train_batch_size}_steps{config.max_steps}"
    unique_model_name = f"{config.experiment_name}_{model_suffix}"
    full_output_dir = f"{config.output_dir}/{unique_model_name}"
    
    # Log all parameters to MLflow
    mlflow.log_params({
        "model_name": config.model_name,
        "language": config.language,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "max_steps": config.max_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "fp16": config.fp16,
        "eval_strategy": config.eval_strategy,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "predict_with_generate": config.predict_with_generate,
        "generation_max_length": config.generation_max_length,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "unique_model_name": unique_model_name
    })
    
    print(f" Training Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Unique Name: {unique_model_name}")
    print(f"   Batch Size: {config.per_device_train_batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Max Steps: {config.max_steps}")
    print(f"   Warmup Steps: {config.warmup_steps}")
    print(f"   Evaluation Strategy: {config.eval_strategy}")
    print(f"   Output Dir: {full_output_dir}")

    try:
        # Load model and processor
        model, processor = load_training_models(config.model_name, config.language)
        
        # Create compute_metrics function
        compute_metrics = create_compute_metrics_function(processor)

        # Create training arguments with all reference parameters
        training_args = create_training_arguments(config, full_output_dir)

        # Create trainer
        trainer = create_trainer(training_args, model, processor, dataset, compute_metrics)
        
        print("Starting training...")
        
        # Track training start time
        import time
        start_time = time.time()
        
        # Train the model
        trainer.train()
        
        # Calculate and log training time
        training_time = (time.time() - start_time) / 60  # minutes
        mlflow.log_metric("training_time_minutes", training_time)
        
        # Log final metrics
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            if "eval_wer" in final_log:
                mlflow.log_metric("final_wer", final_log["eval_wer"])
            if "train_loss" in final_log:
                mlflow.log_metric("final_train_loss", final_log["train_loss"])
        
        # Save model 
        model_path = f"{full_output_dir}/final_model"
        save_model_and_processor(trainer, processor, model_path)
        
        # Log model as MLflow artifact
        mlflow.log_artifacts(full_output_dir, artifact_path="model")
        
        print(f"Training completed!")
        print(f"   Model: {unique_model_name}")
        print(f"   Time: {training_time:.1f} minutes")
        print(f"   Path: {model_path}")
        
        return model_path
        
    except Exception as e:
        # Log error to MLflow
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        print(f"Training failed: {e}")
        raise