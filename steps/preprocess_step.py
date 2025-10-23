from zenml import step
from datasets import DatasetDict
from src.data_preprocessor import (
    load_preprocessing_models,
    create_prepare_dataset_function,
    resample_audio_column
)

@step
def preprocess_step(dataset: DatasetDict, model_name: str = "openai/whisper-small", language: str = "Amharic") -> DatasetDict:
    
    print(f"Loading preprocessing models: {model_name} ({language})...")
    
    # Load models
    feature_extractor, tokenizer = load_preprocessing_models(model_name, language)
    
    print("Resampling audio to 16kHz...")
    
    # Resample audio 
    dataset = resample_audio_column(dataset, target_sampling_rate=16000)

    print("Creating dataset preparation function...")

    # Create prepare function 
    prepare_dataset = create_prepare_dataset_function(feature_extractor, tokenizer)

    print("Processing dataset...")

    # Apply preprocessing
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=1
    )

    print("Preprocessing complete!")
    return dataset