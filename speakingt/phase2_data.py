"""Pinned, audited official-train WAVs with Baseline 01's exact speaker split.

Official test metadata is audited, but no official-test audio is downloaded for
Phase 2 training. Existing GOPT runs and notebooks are read-only inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs/gopt/run-20261102-014018-444335'
DATA = ROOT / 'data/phase2'
TARGETS = ('accuracy', 'fluency', 'prosody', 'total')
REVISION = '613968e3b0b789fc33936fb5eba1973176ba7d11'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def read_audio(path):
    """Require original PCM16/mono/16kHz; retain all samples and silence."""
    with wave.open(str(path), 'rb') as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError(f'Unexpected official WAV format: {path}')
        n = stream.getnframes()
        pcm = stream.readframes(n)
    if len(pcm) != n * 2 or n < 400 or not any(pcm):
        raise ValueError(f'Truncated/empty WAV: {path}')
    return np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768., hashlib.sha256(pcm).hexdigest()


def check_split(split, speakers, official_test_ids=()):
    a, b = split['train_ids'], split['validation_ids']
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
        raise ValueError('Duplicate or overlapping development utterance IDs')
    if (set(a) | set(b)) & set(official_test_ids):
        raise ValueError('Official test contamination')
    if {speakers[i] for i in a} & {speakers[i] for i in b}:
        raise ValueError('Speaker leakage')


def preserve_baseline(baseline=BASELINE, output=ROOT/'experiments/phase2/baseline01.json'):
    baseline, output = Path(baseline), Path(output)
    inventory = {p.relative_to(baseline).as_posix(): sha256(p)
                 for p in sorted(baseline.rglob('*')) if p.is_file()}
    required = ('best.pt', 'config.json', 'split.json', 'summary.json', 'final-test.json', 'metrics-030.json')
    if any(x not in inventory for x in required):
        raise ValueError('Baseline 01 incomplete')
    current = {'name': 'Baseline 01', 'run': baseline.relative_to(ROOT).as_posix(),
               'files': inventory, 'config': json.loads((baseline/'config.json').read_text(encoding='utf-8')),
               'summary': json.loads((baseline/'summary.json').read_text(encoding='utf-8')),
               'historical_test_metrics': json.loads((baseline/'final-test.json').read_text(encoding='utf-8')),
               'test_policy': 'historical only; never used for new model selection'}
    if output.exists():
        prior = json.loads(output.read_text(encoding='utf-8'))
        if prior['files'] != inventory:
            raise ValueError('Protected baseline changed since preservation')
        return prior
    backup = ROOT/'models/baseline01-preserved'
    backup.mkdir(parents=True, exist_ok=True)
    for name in required:
        target = backup/name
        if target.exists() and sha256(target) != inventory[name]:
            raise ValueError('Existing baseline backup differs')
        if not target.exists():
            shutil.copyfile(baseline/name, target)
        if sha256(target) != inventory[name]:
            raise ValueError('Baseline backup hash mismatch')
    write_json(output, current)
    return current


def audit_metadata(baseline=BASELINE, output=DATA):
    from scripts.prepare_gopt import audit_partition, read_mapping
    from .gopt_workbench import load_data
    preserve_baseline(baseline)
    assets = ROOT/'data/gopt/assets'
    scores = json.loads((assets/'scores.json').read_text(encoding='utf-8'))
    phones = json.loads((assets/'phn_dict.json').read_text(encoding='utf-8'))
    train_spk, test_spk = (read_mapping(assets/f'{s}_utt2spk') for s in ('train', 'test'))
    train_wav, test_wav = (read_mapping(assets/f'{s}_wav.scp') for s in ('train', 'test'))
    if set(train_spk) != set(train_wav) or set(test_spk) != set(test_wav):
        raise ValueError('Official metadata ID mismatch')
    if len(train_spk) != 2500 or len(test_spk) != 2500 or set(train_spk) & set(test_spk):
        raise ValueError('Unexpected official partition')
    if set(train_spk.values()) & set(test_spk.values()):
        raise ValueError('Official speaker overlap')
    # Audit all original labels; no model evaluation or model choice uses test.
    for prefix, spk in (('tr', train_spk), ('dev', test_spk), ('new_te', test_spk)):
        audit_partition(assets, prefix, scores, phones, spk)
    original = load_data(ROOT/'data/gopt', 'train')
    split = json.loads((Path(baseline)/'split.json').read_text(encoding='utf-8'))
    check_split(split, train_spk, test_spk)
    if len(split['train_ids']) != 2000 or len(split['validation_ids']) != 500:
        raise ValueError('Expected Baseline 01 2000/500 split')
    if set(split['train_ids'] + split['validation_ids']) != set(original['ids']):
        raise ValueError('Baseline membership differs from official train')
    lock_path = ROOT/'experiments/phase2/corpus-files.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    blobs = {v['path']: v for v in lock['files']}
    split_hash = sha256(Path(baseline)/'split.json')
    rows = []
    for group in ('train', 'validation'):
        for uid in split[group+'_ids']:
            score = scores[uid]
            path = train_wav[uid]
            if path not in blobs or path.split('/')[1] != 'SPEAKER'+train_spk[uid]:
                raise ValueError('Official WAV path mismatch')
            rows.append({'id': uid, 'speaker': train_spk[uid], 'split': group,
                         'source_path': path, 'audio_path': 'audio/'+path,
                         'git_blob_sha1': blobs[path]['git_blob_sha1'], 'bytes': blobs[path]['bytes'],
                         'text': score['text'],
                         'targets': [float(score[k]) for k in ('accuracy','fluency','prosodic','total')]})
    by_text = {s: {' '.join(r['text'].lower().split()) for r in rows if r['split']==s}
               for s in ('train','validation')}
    by_blob = defaultdict(set)
    for group, mapping in (('official_train', train_wav), ('official_test', test_wav)):
        for uid, path in mapping.items():
            by_blob[blobs[path]['git_blob_sha1']].add((group, uid))
    repeated = [sorted(v) for v in by_blob.values() if len(v)>1]
    existing = {}
    for folder in ('data/gopt', 'data/speechocean762-pilot'):
        # Windows glob is case-insensitive; iterate once, never double count.
        found = [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix.lower()=='.wav']
        official_ids = {p.stem for p in found if p.stem in train_spk or p.stem in test_spk}
        existing[folder] = {'files':len(found),'official_train_ids':len(official_ids & set(train_spk)),
                            'official_test_ids':len(official_ids & set(test_spk))}
    audit = {'baseline_split_sha256':split_hash, 'corpus_revision':REVISION,
             'corpus_lock_sha256':sha256(lock_path), 'gop_manifest_sha256':sha256(ROOT/'data/gopt/manifest.json'),
             'official_train':2500,'training':2000,'validation':500,'official_test':2500,
             'training_speakers':len({r['speaker'] for r in rows if r['split']=='train'}),
             'validation_speakers':len({r['speaker'] for r in rows if r['split']=='validation'}),
             'speaker_overlap':0, 'utterance_id_overlap':0,
             'all_5000_gop_labels_audited':True, 'original_audio_inventory':existing,
             'train_validation_shared_texts':len(by_text['train'] & by_text['validation']),
             'official_duplicate_git_blobs':repeated,
             'test_policy':'metadata audit only; excluded from Phase 2 features/training/evaluation',
             'units':'four utterance targets 0–10; not phone PCC',
             'target_order':list(TARGETS)}
    path = Path(output)/'manifest.json'
    manifest = {'schema':1, 'audit':audit, 'rows':rows}
    if path.exists():
        prior=json.loads(path.read_text(encoding='utf-8'))
        if prior['rows'] != rows or prior['audit']['baseline_split_sha256'] != split_hash:
            raise ValueError('Existing Phase 2 dataset/split changed')
        return prior
    write_json(path, manifest)
    write_json(Path(output)/'audit-initial.json', audit)
    return manifest


def _ensure_audio(row, output, reuse):
    path=Path(output)/row['audio_path']
    if path.exists():
        raw=path.read_bytes()
    else:
        raw=None
        for candidate in reuse.get(row['id'], []):
            candidate_bytes=candidate.read_bytes()
            if blob_sha(candidate_bytes)==row['git_blob_sha1']:
                raw=candidate_bytes
                break
        if raw is None:
            url=f'https://raw.githubusercontent.com/jimbozhang/speechocean762/{REVISION}/{row["source_path"]}'
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=40) as response:
                        raw=response.read(row['bytes']+1)
                    break
                except OSError:
                    if attempt==2: raise
                    time.sleep(attempt+1)
        if len(raw)!=row['bytes'] or blob_sha(raw)!=row['git_blob_sha1']:
            raise ValueError('Official source integrity failure: '+row['id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial=path.with_suffix('.partial')
        partial.write_bytes(raw)
        partial.replace(path)
    if len(raw)!=row['bytes'] or blob_sha(raw)!=row['git_blob_sha1']:
        raise ValueError('Cached audio changed: '+row['id'])
    samples, pcm_hash=read_audio(path)
    return {'id':row['id'], 'split':row['split'], 'sha256':hashlib.sha256(raw).hexdigest(),
            'pcm_sha256':pcm_hash,'samples':len(samples),'duration_seconds':len(samples)/16000,
            'path':row['audio_path'], 'git_blob_sha1':row['git_blob_sha1']}


def prepare_audio(output=DATA, workers=8, limit=None):
    output=Path(output)
    manifest=audit_metadata(output=output)
    if limit is not None and (not isinstance(limit,int) or limit<1):
        raise ValueError('limit must be positive or None')
    rows=manifest['rows'] if limit is None else manifest['rows'][:limit]
    reuse=defaultdict(list)
    for folder in (ROOT/'data/speechocean762-pilot', ROOT/'data/gopt/examples'):
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower()=='.wav': reuse[p.stem].append(p)
    start=time.perf_counter()
    done=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_ensure_audio, row, output, reuse) for row in rows]
        for future in as_completed(futures):
            done.append(future.result())
            if len(done)%100==0 or len(done)==len(rows):
                print(f'WAV verified {len(done)}/{len(rows)} ({time.perf_counter()-start:.1f}s)', flush=True)
    group=defaultdict(list)
    for r in done: group[r['pcm_sha256']].append(r)
    cross=[v for v in group.values() if len({r['split'] for r in v})>1]
    if cross: raise ValueError('Exact decoded audio crosses train/validation split')
    report={'complete':len(done)==2500, 'files':sorted(done,key=lambda r:r['id']),
            'count':len(done),'duration_seconds':sum(r['duration_seconds'] for r in done),
            'decoded_duplicate_groups':[[r['id'] for r in v] for v in group.values() if len(v)>1],
            'cross_split_decoded_duplicates':0,'preprocessing':'original PCM16 mono 16kHz; no trim/resample/denoise',
            'elapsed_seconds':time.perf_counter()-start}
    write_json(output/('audio-audit.json' if limit is None else f'audio-audit-first-{limit}.json'),report)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    print(json.dumps({k:v for k,v in prepare_audio(workers=args.workers,limit=args.limit).items() if k!='files'},indent=2))
