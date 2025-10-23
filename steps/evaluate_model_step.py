from zenml import step
from datasets import DatasetDict
from typing import Dict, Optional
from src.evaluation import load_evaluation_models, run_evaluation

@step
def evaluate_model_step(
    model_path: str, 
    dataset: DatasetDict,
    max_samples: Optional[int] = None,
    show_examples: int = 5
) -> Dict[str, float]:
    
    # Load models
    model, processor, device = load_evaluation_models(model_path)

    # evaluation
    results = run_evaluation(
        model=model,
        processor=processor, 
        device=device,
        dataset=dataset,
        max_samples=max_samples,
        show_examples=show_examples
    )
    
    return {
        "wer": results["wer"],
        "samples_evaluated": results["samples_evaluated"]
    }