import os
import requests
import argparse
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_from_url(url, save_path):
    """Download a file from a given URL and save it locally."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("Download failed!")
    else:
        print(f"File downloaded to: {save_path}")

def download_from_hf(repo_id, filename, save_path):
    """Download a file from Hugging Face Hub."""
    print(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(save_path), local_dir_use_symlinks=False)
        print(f"File downloaded to: {save_path}")
    except Exception as e:
        print(f"Download failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Automatically download model checkpoints")
    parser.add_argument("--source", type=str, required=True, choices=["hf", "url"], help="Download source: hf (Hugging Face Hub) or url (custom URL)")
    parser.add_argument("--repo_id", type=str, help="Hugging Face model repository ID (e.g., google/bert-base-uncased)")
    parser.add_argument("--filename", type=str, help="Filename in the Hugging Face repository")
    parser.add_argument("--url", type=str, help="Custom download URL")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the file (including filename)")
    args = parser.parse_args()

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.source == "hf":
        if not args.repo_id or not args.filename:
            print("Please provide a Hugging Face repository ID and filename!")
            return
        download_from_hf(args.repo_id, args.filename, args.save_path)
    elif args.source == "url":
        if not args.url:
            print("Please provide a download URL!")
            return
        download_from_url(args.url, args.save_path)

if __name__ == "__main__":
    main()