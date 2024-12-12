import argparse
from huggingface_hub import HfApi

def create_and_upload_model(hub_model_id, output_dir, hub_token):
    print(f"Uploading model: {hub_model_id} to {output_dir} ")
    api = HfApi()
    repo_details = api.create_repo(
        repo_id=hub_model_id,
        exist_ok=True,
        token=hub_token
    )
    repo_id = repo_details.repo_id
    api.upload_folder(
        commit_message="Saving model from local file system",
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hub_token
    )

def main(args):
    create_and_upload_model(args.hub_model_id, args.output_dir, args.hub_token)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload a model to Hugging Face Hub')
    parser.add_argument('--hub_model_id', type=str, required=True, help='Model ID on the Hugging Face Hub')
    parser.add_argument('--output_dir', type=str, required=True, help='Local directory path where the model files are stored')
    parser.add_argument('--hub_token', type=str, required=True, help='Hugging Face authorization token')

    args = parser.parse_args()
    main(args)
