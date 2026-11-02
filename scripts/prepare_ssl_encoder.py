"""Download only pinned official WavLM files, checking upstream hashes."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speakingt.phase2_data import ROOT, sha256, blob_sha


def prepare_encoder():
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
    from huggingface_hub import hf_hub_download
    lock=json.loads((ROOT/'experiments/phase2/encoder-lock.json').read_text(encoding='utf-8'))
    folder=ROOT/'.cache/ssl-models/wavlm-base-plus'/lock['revision']
    for item in lock['files']:
        path=folder/item['file']
        if not path.exists():
            hf_hub_download(lock['repository'], item['file'], revision=lock['revision'],
                            local_dir=folder, token=False)
        if path.stat().st_size!=item['bytes']:
            raise ValueError('Model asset size mismatch: '+item['file'])
        actual=sha256(path) if 'sha256' in item else blob_sha(path.read_bytes())
        if actual != item.get('sha256',item.get('git_blob_sha1')):
            raise ValueError('Model asset hash mismatch: '+item['file'])
        print('Verified encoder asset:',item['file'],flush=True)
    return folder


if __name__=='__main__':
    print(prepare_encoder())
