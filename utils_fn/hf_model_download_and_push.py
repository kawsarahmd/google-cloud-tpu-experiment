from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import os

def download_and_push_model(source_model, destination_repo, token):

    tokenizer = AutoTokenizer.from_pretrained(source_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(source_model,from_flax=True)
    
    print(f"Pushing to {destination_repo}...")
    model.push_to_hub(destination_repo, token=token)
    tokenizer.push_to_hub(destination_repo, token=token)
    
    print(f"Successfully pushed {source_model} to {destination_repo}")

def main():
    
    token = "HF_TOKEN"
    
    model_mapping = {
        "kawsarahmd/bnT5_base_v1_32k_vocab": "tensorlabco/bnT5_32k",
        "kawsarahmd/bnT5_base_v1_64k_vocab": "tensorlabco/bnT5_64k"
    }
    
    for source_model, destination_repo in model_mapping.items():
        try:
            download_and_push_model(source_model, destination_repo, token)
        except Exception as e:
            print(f"Error processing {source_model}: {str(e)}")

if __name__ == "__main__":
    main()
