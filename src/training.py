import torch
import evaluate
from config.training_config import TrainingConfig
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding

def load_training_models(model_name: str, language: str):
    """Load Whisper model and processor for training"""
    print(f"Loading Whisper model: {model_name} ({language})...")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task="transcribe"
    )
    
    # Configure model for generation
    model.generation_config.language = language.lower()
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    # Move to GPU 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, processor

def create_compute_metrics_function(processor):
    """Create compute_metrics function for WER calculation"""
    metric = evaluate.load("wer")
    tokenizer = processor.tokenizer
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    return compute_metrics

def create_training_arguments(config: TrainingConfig, output_dir: str):
    """Create training arguments from configuration object"""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        eval_strategy=config.eval_strategy,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=False,
        no_cuda=False,
    )

def create_data_collator(processor, model):
    """Create data collator for speech-to-text training"""
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

def create_trainer(training_args, model, processor, dataset, compute_metrics):
    """Create Seq2SeqTrainer with all components"""
    data_collator = create_data_collator(processor, model)
    
    return Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

def save_model_and_processor(trainer, processor, model_path: str):
    """Save trained model and processor to specified path"""
    print(f"Saving model to {model_path}...")
    trainer.save_model(model_path)
    processor.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
