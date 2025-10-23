import torch
import evaluate
from tqdm import tqdm
from datasets import Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, List, Tuple

def load_evaluation_models(model_path: str):
    """Load trained Whisper model and processor for evaluation"""
    print(f"Loading model from {model_path}...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    print(f"Model loaded on {device}")
    return model, processor, device

def run_evaluation(model, processor, device, dataset, max_samples: int = None, show_examples: int = 3):
    test_samples = dataset["test"].cast_column("audio", Audio(sampling_rate=16000))
    
    # Prepare samples
    if max_samples is None:
        eval_samples = test_samples
        print(f"Evaluating on ALL {len(test_samples)} test samples...")
    else:
        eval_samples = test_samples.select(range(min(max_samples, len(test_samples))))
        print(f"Evaluating on {len(eval_samples)} samples...")
    
    predictions, references = [], []
    wer_metric = evaluate.load("wer")
    
    # Evaluation loop 
    for sample in tqdm(eval_samples, desc="Processing"):
        input_features = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
        references.append(sample["sentence"])
    
    # Calculate WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Log results
    print(f"\nEvaluation Results:")
    print(f"   WER: {wer * 100:.2f}%")
    print(f"   Samples evaluated: {len(predictions)}")
    
    # Show examples
    print(f"\nSample Results (showing {min(show_examples, len(predictions))}):")
    for i in range(min(show_examples, len(predictions))):
        print(f"   {i+1}. Reference: {references[i]}")
        print(f"      Prediction: {predictions[i]}")
        print()
    
    return {
        "wer": wer * 100,
        "samples_evaluated": len(predictions),
        "predictions": predictions,
        "references": references
    }