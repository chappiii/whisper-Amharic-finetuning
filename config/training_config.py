from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "openai/whisper-small"
    language: str = "Amharic"
    
    # Training parameters  
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 2000
    gradient_checkpointing: bool = False
    fp16: bool = True
    
    # Evaluation parameters
    eval_strategy: str = "steps"
    per_device_eval_batch_size: int = 4
    predict_with_generate: bool = True
    generation_max_length: int = 225
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Experiment tracking
    experiment_name: str = "whisper_experiment"
    output_dir: str = "./checkpoints"

# Default configuration
DEFAULT_CONFIG = TrainingConfig()