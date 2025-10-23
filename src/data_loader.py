from datasets import load_dataset, DatasetDict

def load_common_voice_amharic():
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "am",
        split="train+validation", trust_remote_code=True
    )
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_17_0", "am",
        split="test", trust_remote_code=True
    )

    # remove unnecessary columns
    common_voice = common_voice.remove_columns([
        "accent", "age", "client_id", "down_votes", "gender",
        "locale", "path", "segment", "up_votes"
    ])
    return common_voice
