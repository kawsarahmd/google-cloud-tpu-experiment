"""
Script to merge two Bangla news summary datasets from Hugging Face:
1. kawsarahmd/bangla_news_summary_24k
2. kawsarahmd/xlsum_bangla

The merged dataset is then split into 90% train, 5% validation, 5% test
and pushed back to Hugging Face.
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Configuration
HF_TOKEN = ""  
OUTPUT_REPO = "kawsarahmd/merged_24kbansum_xlsum_summary"  


def load_and_prepare_datasets():
    """Load both datasets from Hugging Face"""
    print("Loading datasets from Hugging Face...")
    
    # Load dataset 1: bangla_news_summary_24k
    dataset1 = load_dataset("kawsarahmd/bangla_news_summary_24k")
    print(f"Dataset 1 loaded: {dataset1}")
    
    # Load dataset 2: xlsum_bangla
    dataset2 = load_dataset("kawsarahmd/xlsum_bangla")
    print(f"Dataset 2 loaded: {dataset2}")
    
    return dataset1, dataset2

def normalize_dataset1(dataset1):
    """
    Normalize dataset1 to match the target schema.
    Original: id, url, category, title, content, content_summary
    Target: id, url, title, summary, text
    """
    print("Normalizing dataset1...")
    
    def rename_columns(example):
        return {
            "id": example["id"],
            "url": example["url"],
            "title": example["title"],
            "summary": example["content_summary"],
            "text": example["content"],
        }
    
    # Apply transformation to all splits
    normalized_dataset1 = {}
    for split in dataset1.keys():
        normalized_dataset1[split] = dataset1[split].map(
            rename_columns,
            remove_columns=["category", "content", "content_summary", "__index_level_0__"]
        )
    
    return normalized_dataset1

def normalize_dataset2(dataset2):
    """
    Dataset2 already has the correct schema: id, url, title, summary, text
    Just ensure all required columns are present
    """
    print("Normalizing dataset2...")
    
    def ensure_columns(example):
        return {
            "id": example["id"],
            "url": example["url"],
            "title": example["title"],
            "summary": example["summary"],
            "text": example["text"],
        }
    
    normalized_dataset2 = {}
    for split in dataset2.keys():
        normalized_dataset2[split] = dataset2[split].map(ensure_columns)
    
    return normalized_dataset2

def clean_dataset(df, dataset_name=""):
    """
    Clean dataset by removing rows with empty or null values
    in critical columns: id, url, title, summary, text
    """
    print(f"\nCleaning {dataset_name}...")
    print(f"  Initial rows: {len(df)}")
    
    initial_size = len(df)
    
    # Define critical columns
    critical_columns = ["id", "url", "title", "summary", "text"]
    
    # Count missing values before cleaning
    missing_before = df[critical_columns].isnull().sum()
    print(f"  Missing values before cleaning:")
    for col in critical_columns:
        if missing_before[col] > 0:
            print(f"    - {col}: {missing_before[col]}")
    
    # Remove rows with null values in critical columns
    df_cleaned = df.dropna(subset=critical_columns)
    
    # Remove rows where critical columns are empty strings or whitespace
    for col in critical_columns:
        df_cleaned = df_cleaned[df_cleaned[col].astype(str).str.strip() != ""]
    
    rows_removed = initial_size - len(df_cleaned)
    print(f"  Rows removed: {rows_removed}")
    print(f"  Final rows: {len(df_cleaned)}")
    
    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def merge_and_split_datasets(dataset1_dict, dataset2_dict, equal_distribution=True, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05):
    """
    Merge datasets and create new splits with optional equal distribution of sources.
    
    Args:
        dataset1_dict: Normalized dataset 1 dictionary
        dataset2_dict: Normalized dataset 2 dictionary
        equal_distribution: If True, mix both datasets evenly in all splits (shuffled together)
                          If False, concatenate as-is (Dataset 1 then Dataset 2)
        train_ratio: Ratio for training set (default: 0.90)
        val_ratio: Ratio for validation set (default: 0.05)
        test_ratio: Ratio for test set (default: 0.05)
    
    Note: 
        - ALL data from both datasets is kept (no data is discarded)
        - With equal_distribution=True, both datasets are shuffled together for even mixing
        - Final dataset will have ~34k examples (24k + 10k) minus cleaned rows
    """
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)
    
    # Combine all splits from both datasets
    all_data1 = []
    all_data2 = []
    
    for split in dataset1_dict.keys():
        all_data1.append(dataset1_dict[split])
    
    for split in dataset2_dict.keys():
        all_data2.append(dataset2_dict[split])
    
    # Concatenate datasets by source
    combined_dataset1 = concatenate_datasets(all_data1)
    combined_dataset2 = concatenate_datasets(all_data2)
    
    print(f"Dataset 1 (kawsarahmd/bangla_news_summary_24k) - Total examples: {len(combined_dataset1)}")
    print(f"Dataset 2 (kawsarahmd/xlsum_bangla) - Total examples: {len(combined_dataset2)}")
    
    # Convert to pandas for cleaning
    df1 = combined_dataset1.to_pandas()
    df2 = combined_dataset2.to_pandas()
    
    # Clean datasets
    df1_cleaned = clean_dataset(df1, "Dataset 1")
    df2_cleaned = clean_dataset(df2, "Dataset 2")
    
    # Add source column to track which dataset each row came from
    df1_cleaned['_source'] = 'dataset1'
    df2_cleaned['_source'] = 'dataset2'
    
    # Merge both datasets - KEEP ALL DATA
    merged_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)
    
    print("\n" + "-"*60)
    print("MERGING DATASETS (KEEPING ALL DATA)")
    print("-"*60)
    print(f"Dataset 1 (kept): {len(df1_cleaned)} examples")
    print(f"Dataset 2 (kept): {len(df2_cleaned)} examples")
    
    # Shuffle the merged dataset
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if equal_distribution:
        print("✓ Equal distribution enabled: Both datasets will be mixed evenly in all splits")
    
    print("\n" + "-"*60)
    print("MERGED DATASET STATISTICS")
    print("-"*60)
    print(f"Total merged examples: {len(merged_df)}")
    
    # Calculate split sizes
    total_size = len(merged_df)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Train ratio: {train_ratio*100:.1f}% ({train_size} examples)")
    print(f"Validation ratio: {val_ratio*100:.1f}% ({val_size} examples)")
    print(f"Test ratio: {test_ratio*100:.1f}% ({test_size} examples)")
    
    # Split data
    # First split: separate train from val+test
    train_df, temp_df = train_test_split(
        merged_df, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Second split: split the remaining into val and test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_test_ratio),
        random_state=42
    )
    
    print("\n" + "-"*60)
    print("FINAL SPLIT SIZES")
    print("-"*60)
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    
    # Show distribution by source in each split
    if equal_distribution:
        print("\n" + "-"*60)
        print("DISTRIBUTION BY SOURCE (EVENLY MIXED)")
        print("-"*60)
        
        ds1_train = (train_df['_source'] == 'dataset1').sum()
        ds2_train = (train_df['_source'] == 'dataset2').sum()
        print(f"Train set:")
        print(f"  Dataset 1: {ds1_train} examples ({100*ds1_train/len(train_df):.1f}%)")
        print(f"  Dataset 2: {ds2_train} examples ({100*ds2_train/len(train_df):.1f}%)")
        
        ds1_val = (val_df['_source'] == 'dataset1').sum()
        ds2_val = (val_df['_source'] == 'dataset2').sum()
        print(f"Validation set:")
        print(f"  Dataset 1: {ds1_val} examples ({100*ds1_val/len(val_df):.1f}%)")
        print(f"  Dataset 2: {ds2_val} examples ({100*ds2_val/len(val_df):.1f}%)")
        
        ds1_test = (test_df['_source'] == 'dataset1').sum()
        ds2_test = (test_df['_source'] == 'dataset2').sum()
        print(f"Test set:")
        print(f"  Dataset 1: {ds1_test} examples ({100*ds1_test/len(test_df):.1f}%)")
        print(f"  Dataset 2: {ds2_test} examples ({100*ds2_test/len(test_df):.1f}%)")
    
    # Remove _source column before returning
    train_df = train_df.drop('_source', axis=1)
    val_df = val_df.drop('_source', axis=1)
    test_df = test_df.drop('_source', axis=1)
    
    # Convert back to datasets
    from datasets import Dataset
    
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    # Create DatasetDict
    merged_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return merged_dataset

def push_to_hub(dataset_dict, repo_name, token):
    """Push the merged dataset to Hugging Face Hub"""
    print(f"Pushing dataset to Hugging Face Hub as '{repo_name}'...")
    
    try:
        dataset_dict.push_to_hub(repo_name, token=token, private=False)
        print(f"✓ Successfully pushed to: https://huggingface.co/datasets/{repo_name}")
        return True
    except Exception as e:
        print(f"✗ Error pushing to hub: {str(e)}")
        return False

def save_locally(dataset_dict, output_dir="./merged_bangla_dataset"):
    """Save the dataset locally as an alternative"""
    print(f"Saving dataset locally to {output_dir}...")
    dataset_dict.save_to_disk(output_dir)
    print(f"✓ Dataset saved to {output_dir}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("Bangla News Summary Dataset Merger")
    print("=" * 60)
    
    # Load datasets
    dataset1, dataset2 = load_and_prepare_datasets()
    
    # Normalize to target schema
    normalized_dataset1 = normalize_dataset1(dataset1)
    normalized_dataset2 = normalize_dataset2(dataset2)
    
    # Merge and create new splits
    # Set equal_distribution=True for 50/50 distribution from each dataset
    # Change to equal_distribution=False to use all data from both datasets
    merged_dataset = merge_and_split_datasets(
        normalized_dataset1, 
        normalized_dataset2,
        equal_distribution=True,  # Set to True for equal distribution from both datasets
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05
    )
    
    # Display dataset info
    print("\n" + "=" * 60)
    print("Final Merged Dataset Info:")
    print("=" * 60)
    print(merged_dataset)
    print("\nDataset Features:")
    print(merged_dataset["train"].features)
    
    # Save locally
    print("\n" + "=" * 60)
    save_locally(merged_dataset)
    
    # Push to Hub
    print("\n" + "=" * 60)
    if HF_TOKEN != "your_hugging_face_token_here":
        push_to_hub(merged_dataset, OUTPUT_REPO, HF_TOKEN)
    else:
        print("⚠ Hugging Face token not configured.")
        print("To push to Hub, update HF_TOKEN and OUTPUT_REPO in the script.")
        print("Then uncomment the push_to_hub call below.")
    
    print("\n" + "=" * 60)
    print("✓ Process completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
