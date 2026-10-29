import argparse
import os
import shutil
import sys

sys.path.insert(0, ".")

from huggingface_hub import HfApi
from mlcore import runs
from mlcore.tokenizer import load_tokenizer

from src.config import Config
from src.hf_wrapper import EnCyForTranslation
from src.train import STAGE

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Config name (e.g. base)")
parser.add_argument("--repo", required=True, help="HuggingFace repo (e.g. username/en-cy)")
args = parser.parse_args()

config = runs.load_config_file(f"configs/{args.config}.yaml", Config)

run_path = runs.get_run_path(config)
stage_path = runs.get_stage_path(config, STAGE)
tmp_path = "/tmp/hf_push"
os.makedirs(tmp_path, exist_ok=True)

print(f"Loading model from {stage_path}...")
model = EnCyForTranslation.from_run(stage_path)
tokenizer = load_tokenizer(run_path)

print("Saving HF format...")
model.save_pretrained(tmp_path)
tokenizer.save_pretrained(tmp_path)

shutil.copy("README.md", f"{tmp_path}/README.md")
# For reference only: loading needs this project and mlcore installed
shutil.copy("src/hf_wrapper.py", f"{tmp_path}/hf_wrapper.py")

print(f"Pushing to {args.repo}...")
api = HfApi()
api.create_repo(args.repo, exist_ok=True)
api.upload_folder(folder_path=tmp_path, repo_id=args.repo)

shutil.rmtree(tmp_path)
print(f"Done — https://huggingface.co/{args.repo}")
