from datasets import load_dataset, DatasetDict
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from collections import Counter

def visualize_sample_audio(train_dataset, sample_idx=0):
    """Visualize waveform, spectrogram, and mel spectrogram"""
    print("Audio Visualization - Sample Audio File")
    
    sample = train_dataset[sample_idx]
    audio_array = sample['audio']['array']
    sampling_rate = sample['audio']['sampling_rate']

    plt.figure(figsize=(14, 8))

    # Waveform
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(audio_array)/sampling_rate, len(audio_array))
    plt.plot(time_axis, audio_array)
    plt.title(f'Waveform: "{sample["sentence"]}"')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    # Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.stft(audio_array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Mel Spectrogram (what Whisper uses)
    plt.subplot(3, 1, 3)
    S = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, n_mels=80)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram (Whisper Input Format)')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

    print(f"Audio duration: {len(audio_array)/sampling_rate:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Text: {sample['sentence']}")
    
    return sample

def analyze_text_characters(sentences):
    """Analyze character distribution in Amharic text"""
    print("Amharic Text Character Analysis")

    all_text = ' '.join(sentences)
    char_counts = Counter(all_text)
    common_chars = char_counts.most_common(20)

    plt.figure(figsize=(14, 6))
    chars, counts = zip(*common_chars)
    plt.bar(chars, counts)
    plt.title('Most Common Characters in Amharic Text')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    for i, (char, count) in enumerate(common_chars):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    plt.show()

    print(f"\nCharacter Statistics:")
    print(f"Total characters: {len(all_text):,}")
    print(f"Unique characters: {len(char_counts)}")
    print(f"Most common characters: {common_chars[:10]}")

    # Amharic-specific analysis
    amharic_chars = [c for c in char_counts.keys() if ord(c) >= 4608 and ord(c) <= 4991]  # Amharic Unicode range
    print(f"Amharic script characters: {len(amharic_chars)}")
    
    return char_counts, common_chars, amharic_chars

def load_common_voice_amharic():
    """Load Common Voice Amharic dataset"""
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "am",
        split="train+validation", trust_remote_code=True
    )
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "am",
        split="test", trust_remote_code=True
    )
    return common_voice

def analyze_dataset_overview(dataset):
    """Analyze and display dataset overview information"""
    print(" Dataset Overview")
    print(f"Dataset type: {type(dataset)}")
    print(f"Splits available: {list(dataset.keys())}")

    # Access individual splits
    for split_name, split_data in dataset.items():
        print(f"\n {split_name.upper()} Split:")
        print(f"  Samples: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        print(f"  Columns: {split_data.column_names}")

    # Total samples across all splits
    total_samples = sum(len(split_data) for split_data in dataset.values())
    print(f"\n Total samples across all splits: {total_samples}")
    
    return total_samples

def get_sample_data_info(dataset, split_name="train"):
    """Get detailed information about a train"""
    split_data = dataset[split_name]

    # Sample data point
    print(f"\n Sample train data point:")
    sample = split_data[0]
    for key, value in sample.items():
        if key == "audio":
            print(f"  {key}: array length {len(value['array'])}, sample_rate {value['sampling_rate']}")
        else:
            print(f"  {key}: {value}")
    
    return split_data, sample

def analyze_text_statistics(train_dataset):
    """Analyze text characteristics and return statistics"""
    sentences = [item['sentence'] for item in train_dataset]
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # Create visualization
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(sentence_lengths, bins=50, alpha=0.7)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Words per sentence')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.boxplot(sentence_lengths)
    plt.title('Sentence Length Box Plot')
    plt.ylabel('Words per sentence')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nText Statistics:")
    print(f"Min words: {min(sentence_lengths)}")
    print(f"Max words: {max(sentence_lengths)}")
    print(f"Mean words: {np.mean(sentence_lengths):.1f}")
    print(f"Median words: {np.median(sentence_lengths):.1f}")
    
    return sentences, sentence_lengths

def analyze_audio_statistics(train_dataset, sample_size=100):
    """Analyze audio characteristics and return statistics"""
    print(f"Audio Analysis (sampling {sample_size} files for speed)")

    # Sample data for speed
    sample_data = train_dataset.select(range(min(sample_size, len(train_dataset))))
    audio_durations = []

    for item in sample_data:
        audio_array = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        duration = len(audio_array) / sampling_rate
        audio_durations.append(duration)

    return audio_durations

def plot_audio_analysis(audio_durations, sentence_lengths):
    """Create audio analysis visualizations"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(audio_durations, bins=30, alpha=0.7, color='orange')
    plt.title('Audio Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sample_sentence_lengths = sentence_lengths[:len(audio_durations)]
    plt.scatter(sample_sentence_lengths, audio_durations, alpha=0.6)
    plt.title('Text Length vs Audio Duration')
    plt.xlabel('Words per sentence')
    plt.ylabel('Audio duration (seconds)')

    plt.tight_layout()
    plt.show()

    print(f"\nAudio Statistics:")
    print(f"Min duration: {min(audio_durations):.2f}s")
    print(f"Max duration: {max(audio_durations):.2f}s")
    print(f"Mean duration: {np.mean(audio_durations):.2f}s")
    print(f"Median duration: {np.median(audio_durations):.2f}s")