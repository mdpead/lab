import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, ".")

from huggingface_hub import HfApi

from src.hf_wrapper import EnCyForTranslation
from src.tokenizer import load_tokenizer
from src.utils import get_run_path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Config name (e.g. base)")
parser.add_argument("--repo", required=True, help="HuggingFace repo (e.g. username/en-cy)")
args = parser.parse_args()

with open(f"configs/{args.config}.yaml") as f:
    config = yaml.safe_load(f)

run_path = get_run_path(config)
tmp_path = "/tmp/hf_push"
os.makedirs(tmp_path, exist_ok=True)

print(f"Loading model from {run_path}...")
model = EnCyForTranslation.from_run(run_path)
tokenizer = load_tokenizer(run_path)

print("Saving HF format...")
model.save_pretrained(tmp_path)
tokenizer.save_pretrained(tmp_path)

shutil.copy("README.md", f"{tmp_path}/README.md")
shutil.copy("src/hf_wrapper.py", f"{tmp_path}/hf_wrapper.py")
shutil.copy("src/model.py", f"{tmp_path}/model.py")

print(f"Pushing to {args.repo}...")
api = HfApi()
api.create_repo(args.repo, exist_ok=True)
api.upload_folder(folder_path=tmp_path, repo_id=args.repo)

shutil.rmtree(tmp_path)
print(f"Done — https://huggingface.co/{args.repo}")
