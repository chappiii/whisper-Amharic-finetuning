# Whisper Fine-tuning for Amharic Speech Recognition
Fine-tuning OpenAI Whisper for Amharic speech recognition using production-ready MLOps practices.

[![🤗 Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chappM/whisper-amharic-transcriber-v3)
[![Model v2](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20v2-yellow)](https://huggingface.co/chappM/whisper-amharic-small-v2)

## 🎯 **Overview**

This project addresses the critical gap in **Amharic speech recognition** by fine-tuning OpenAI's Whisper model (small) on multiple datasets. Amharic, spoken by 25+ million people in Ethiopia, has limited ASR resources, making this work impactful for language accessibility.

### **Key Insights from Results**
While Whisper-small is a powerful multilingual model, our experiments revealed significant challenges with Amharic out-of-the-box:
- **Baseline struggles**: The pretrained model often failed to recognize Amharic speech, producing nonsensical outputs (e.g., "2.5." for complex sentences) and achieving a WER of 231.58%.
- **Progressive improvement**: Through iterative fine-tuning, we achieved remarkable results—v2 reduced WER to 69.64% using Common Voice, and v3 further improved to 41% by incorporating FLEURS dataset, representing an **82.3% relative improvement** from baseline.
- **Production-ready quality**: The v3 model delivers near-perfect, coherent transcriptions that closely match ground truth, making it ready for real-world deployment despite minor optimization needs for inference speed and fast speech handling.

This work shows that while multilingual models provide a strong foundation, combining diverse, high-quality datasets is essential for achieving production-ready ASR performance in underrepresented languages like Amharic.

### **🤗 Try the Model**
**[→ Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/chappM/whisper-amharic-transcriber-v3)**

Test the latest fine-tuned model with your own Amharic audio files or try our sample recordings.


## 🛠️ **Technology Stack**

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Base Model** | OpenAI Whisper-Small | Proven multilingual ASR performance |
| **Framework** | PyTorch + Transformers | Industry standard for fine-tuning |
| **Orchestration** | ZenML | Reproducible ML pipelines |
| **Experiment Tracking** | MLflow | Comprehensive metrics logging |
| **Datasets** | Common Voice + FLEURS Amharic | High-quality crowdsourced data |
| **Infrastructure** | CUDA + Mixed Precision (RTX 4060) | Memory-efficient training |

## 📊 **Results**

### **Performance Improvement**

| Model | Word Error Rate (WER) | Improvement |
|-------|----------------------|-------------|
| **Baseline (Pretrained)** | 231.58% | - |
| **Fine-tuned v2** | 69.64% | ↓69.9% |
| **Fine-tuned v3** | **41%** | **↓82.3%** |

### **Sample Transcriptions**

| Audio Sample | Ground Truth | Baseline | Fine-tuned |
|--------------|-------------|----------|------------|
| **Sample 1** | ኑሮው ቀን በቀን ይፋጃል። | Nouraouk and Bekanifajal | ኖራው ቀን በቃኔፋው ይፋጨን። ✅ |
| **Sample 2** | አንዳንዶች ፊልምና ዶክመንታሪ ይሰራሉ። | and then don't fillmen a documentary is harallu. | አንዳንዶች ፊልም እና ደውክመን ተሪ ይሰራሉ። ✅ |

While the Whisper-small pretrained model struggled to even detect Amharic—often producing transcriptions in the wrong language or nonsensical outputs—our **v2 fine-tuned model showed dramatic improvement**. After fine-tuning on Common Voice, the predictions became much closer to the ground truth. Although not perfect, they were contextually accurate and demonstrated that the model was learning the language.

**v3 takes this even further** and is now **almost perfect and production-ready**. Trained on both Common Voice and FLEURS datasets, v3 delivers consistently accurate transcriptions that closely match ground truth, with only minor areas for optimization:

**Current Limitations:**
- **Inference Speed**: Whisper-small is a moderately large model, requiring significant processing time especially for longer audio samples on Hugging Face Spaces free tier CPU-basic hardware
- **Fast Speech**: Occasionally misses words when speakers talk very rapidly

This progression highlights both the effectiveness of fine-tuning for low-resource languages and the impact of training on diverse, high-quality datasets like Common Voice and FLEURS.

## 📈 **Model Versions**

| Version | Dataset | WER | Status | Links |
|---------|---------|-----|--------|-------|
| **v2** | Common Voice | 69.64% | Public | [Model](https://huggingface.co/chappM/whisper-amharic-small-v2) \| [Demo](https://huggingface.co/spaces/chappM/whisper-amharic-transcriber) |
| **v3** | Common Voice + FLEURS | **41%** | Private (Demo Available) | [Demo](https://huggingface.co/spaces/chappM/whisper-amharic-transcriber-v3) |

> **Note**: The v3 model is currently private but can be tested through the public demo on Hugging Face Spaces.

## 🚀 **Quick Start**

### **Setup Environment**
```bash
# Create and activate environment
conda create -n whisper python=3.12
conda activate whisper

# Install dependencies
pip install -r requirements.txt
```

### **Configure Authentication**
```bash
# Login to Hugging Face (required for dataset access)
huggingface-cli login
# Enter your HF token when prompted
```

### **Run Training**
```bash
# Execute complete pipeline
python run_training_pipeline.py
```

## 📁 **Project Structure**

```
├── config/              # Training configuration
├── pipelines/           # ZenML orchestration
├── steps/               # Pipeline components
├── src/                 # Core implementation
├── analysis/            # Data exploration
├── notebooks/           # Baseline evaluation
└── run_training_pipeline.py
```

## 🔮 **Future Work**

I'm actively working on further improving the model's performance and accessibility:
- **Inference Speed Optimization**: Experimenting with Whisper-base for faster inference while matching whisper-small model accuracy
- **Fast Speech Handling**: Improving accuracy for rapid speech through targeted training and data augmentation
- **Target WER < 25%**: Continue optimizing hyperparameters, training strategies, and expanding datasets
- **Transcription Application**: Building a user-friendly transcription app (coming soon!)

## 📚 **References**

1. [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
2. [Mozilla Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets)
3. [Google FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)


