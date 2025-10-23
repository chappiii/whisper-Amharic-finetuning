from transformers import WhisperFeatureExtractor, WhisperTokenizer
from datasets import Audio

def load_preprocessing_models(model_name: str, language: str):
    """Load feature extractor and tokenizer for preprocessing"""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=language,
        task="transcribe"
    )
    return feature_extractor, tokenizer

def create_prepare_dataset_function(feature_extractor, tokenizer):
    """Create the prepare_dataset function with loaded models"""
    def prepare_dataset(batch):
        """Prepare batch for training - convert audio to features and text to tokens"""
        audio = batch["audio"]
        
        # Convert audio to mel spectrograms
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Convert text to token IDs
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        
        return batch
    
    return prepare_dataset

def resample_audio_column(dataset, target_sampling_rate: int = 16000):
    """Resample audio to target sampling rate"""
    return dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))