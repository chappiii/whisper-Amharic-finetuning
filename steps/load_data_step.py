from zenml import step
from src.data_loader import load_common_voice_amharic

@step
def load_data_step():
    dataset = load_common_voice_amharic()
    return dataset
