from zenml import pipeline
from config.training_config import TrainingConfig, DEFAULT_CONFIG
from steps.load_data_step import load_data_step
from steps.preprocess_step import preprocess_step
from steps.train_model_step import train_model_step
from steps.evaluate_model_step import evaluate_model_step

@pipeline
def whisper_training_pipeline(config: TrainingConfig = DEFAULT_CONFIG):
    
    # Load data
    original_dataset = load_data_step()

    # Preprocess data
    preprocessed_dataset = preprocess_step(original_dataset)
    
    # Train model
    model_path = train_model_step(dataset=preprocessed_dataset, config=config)
    
    # Evaluate the trained model
    results = evaluate_model_step(model_path=model_path, dataset=original_dataset)

    return results