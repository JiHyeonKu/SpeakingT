"""Read-only audit of existing artifacts; writes only new Phase 3 reports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt.phase2_data import ROOT, sha256, write_json
from speakingt.ssl_workbench import score_report, load_manifest


def inventory():
    paths=set()
    for name in ('runs/gopt','runs/ssl','data','models','.cache/ssl-models','reports/phase2'):
        paths.update(p for p in (ROOT/name).rglob('*') if p.is_file())
    for name in ('notebooks/Phase2_Frozen_WavLM.ipynb','notebooks/SpeakingT_GOPT.ipynb','start-phase2.cmd'):
        paths.add(ROOT/name)
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}


def audit():
    out=ROOT/'reports/phase3/preservation-before.json'
    if out.exists(): raise ValueError('Preservation snapshot already exists')
    write_json(out,inventory())
    manifest=load_manifest()
    expected=[r['id'] for r in manifest['rows'] if r['split']=='validation']
    report=ROOT/'reports/phase2/comparison-20261102-103810-165030'
    summary=json.loads((report/'summary.json').read_text(encoding='utf-8'))
    records=[]; predictions=[]
    with np.load(report/'gopt-validation-predictions.npz') as p:
        ids=p['ids'].copy(); targets=p['targets'].copy(); speakers=p['speakers'].copy(); gopt=p['predictions'].copy()
    assert ids.tolist()==expected
    for r in summary['records']:
        path=ROOT/r['run']; cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
        assert cfg['hidden']==128 and cfg['dropout']==.5
        with np.load(path/'best-validation-predictions.npz') as p:
            assert all(np.array_equal(p[k],v) for k,v in [('ids',ids),('targets',targets),('speakers',speakers)])
            predictions.append(p['predictions'].copy())
        records.append({'run':r['run'],'seed':cfg['seed'],'checkpoint_sha256':sha256(path/'best.pt'),
                        'validation':score_report(predictions[-1],targets)})
    assert sorted(r['seed'] for r in records)==[7,19,42]
    ensemble=np.stack(predictions).mean(0)
    difference=((ensemble-targets)**2-(gopt-targets)**2).mean(1)
    groups=[np.flatnonzero(speakers==s) for s in np.unique(speakers)]
    rng=np.random.default_rng(20261102)
    delta=[float(difference[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])].mean()) for _ in range(3000)]
    small=[]
    for path in sorted((ROOT/'runs/ssl').glob('notebook-full-*/config.json')):
        cfg=json.loads(path.read_text(encoding='utf-8'))
        if cfg['hidden']==32:
            small.append({'run':path.parent.relative_to(ROOT).as_posix(),'seed':cfg['seed'],'dropout':cfg['dropout'],
                          'summary':json.loads((path.parent/'summary.json').read_text(encoding='utf-8'))})
    result={'dropout_05_runs':records,'hidden_32_runs':small,'ensemble':score_report(ensemble,targets),
            'gopt':score_report(gopt,targets),'macro_mse_delta':float(difference.mean()),
            'speaker_bootstrap_95pct':np.percentile(delta,[2.5,97.5]).tolist(),'replicates':3000,'seed':20261102,
            'source_report':report.relative_to(ROOT).as_posix(),'test_used':False,'protected_files':len(inventory())}
    write_json(ROOT/'experiments/phase3/phase2-audit.json',result)
    print(json.dumps({k:result[k] for k in ('macro_mse_delta','speaker_bootstrap_95pct','protected_files')},indent=2))


def verify():
    before=json.loads((ROOT/'reports/phase3/preservation-before.json').read_text(encoding='utf-8'))
    changed=[p for p,h in before.items() if not (ROOT/p).exists() or sha256(ROOT/p)!=h]
    if changed: raise ValueError(f'Protected files changed: {changed[:10]}')
    write_json(ROOT/'experiments/phase3/preservation.json',{'protected_files':len(before),'changed_files':0,
               'snapshot_sha256':sha256(ROOT/'reports/phase3/preservation-before.json')})
    print('Protected files unchanged:',len(before))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); a=p.parse_args()
    verify() if a.verify else audit()
