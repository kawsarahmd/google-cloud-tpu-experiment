from datasets import load_dataset, DatasetDict, Dataset

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", keep_in_memory=True)

train_data = dataset["train"].remove_columns([col for col in dataset["train"].column_names if col != "text"])

train_split_percenatge = 0.99
split_index = int(len(train_data) * train_split_percenatge)
train_split = train_data.select(range(split_index))
val_split = train_data.select(range(split_index, len(train_data)))

# Create new DatasetDict
new_dataset = DatasetDict({
    "train": Dataset.from_dict(train_split),
    "validation": Dataset.from_dict(val_split)
})

# Push to the Hugging Face Hub
new_dataset.push_to_hub(
    "tensorlabco/fineweb-edu-sample-10BT",
    token="",
)
