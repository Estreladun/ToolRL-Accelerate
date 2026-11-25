"""Resolve immutable public data/model resources without cloning source repositories."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'configs/resources.json'


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


def offline():
    return os.environ.get('HF_HUB_OFFLINE', '').upper() in {'1', 'TRUE', 'YES', 'ON'}


def complete_model(path):
    path = Path(path)
    if not all((path / f).is_file() for f in ['config.json', 'tokenizer_config.json', 'tokenizer.json']):
        return False
    if (path / 'model.safetensors').is_file():
        return True
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        files = set(json.loads(index.read_text())['weight_map'].values())
        return bool(files) and all((path / x).is_file() for x in files)
    return False


def resolve_model(model, revision=None):
    local = Path(model).expanduser()
    if local.is_dir():
        if not complete_model(local):
            raise ValueError('Expected a complete Qwen-style safetensors checkpoint')
        files = sorted(p for p in local.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.txt'})
        return local.resolve(), dict(model=str(local.resolve()), files={p.name: checksum(p) for p in files})
    revision = revision or catalog()['models'].get(model)
    if not revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('Supply MODEL_REVISION as a full commit SHA for unlisted Hugging Face models')
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    kwargs = dict(repo_id=model, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'])
    try:
        path = Path(snapshot_download(**kwargs, local_files_only=True))
    except LocalEntryNotFoundError:
        path = None
    if path is None or not complete_model(path):
        path = Path(snapshot_download(**kwargs, local_files_only=offline()))
    if not complete_model(path):
        raise RuntimeError(f'Incomplete model snapshot: {path}')
    return path.resolve(), dict(model=model, revision=revision)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--model-revision')
    a = p.parse_args()
    print(resolve_model(a.model, a.model_revision))

if __name__ == '__main__': main()
