"""Development-only seed summary and paired speaker-cluster uncertainty."""
import csv
import json
from pathlib import Path
import numpy as np
from .phase2_data import ROOT, TARGETS, write_json, sha256
from .ssl_workbench import score_report


def paired_bootstrap(a,b,targets,speakers,replicates=1000,seed=20261102):
    """Sample speakers, retaining all their utterances; positive PCC delta favors a."""
    if a.shape!=b.shape or a.shape!=targets.shape or len(speakers)!=len(a):
        raise ValueError('Paired arrays must describe the same observations')
    unique=sorted(set(speakers.tolist()))
    if len(unique)<2: raise ValueError('Need multiple held-out speakers')
    rng=np.random.default_rng(seed)
    groups={speaker:np.flatnonzero(speakers==speaker) for speaker in unique}
    deltas={name:{'mse':[],'pcc':[]} for name in TARGETS}
    for _ in range(replicates):
        indices=np.concatenate([groups[s] for s in rng.choice(unique,len(unique),replace=True)])
        ma=score_report(a[indices],targets[indices])['targets']
        mb=score_report(b[indices],targets[indices])['targets']
        for name in TARGETS:
            for metric in ('mse','pcc'):
                va,vb=ma[name][metric],mb[name][metric]
                if va is not None and vb is not None: deltas[name][metric].append(va-vb)
    def interval(x):
        return {'low':float(np.percentile(x,2.5)),'high':float(np.percentile(x,97.5)),
                'valid_replicates':len(x)} if x else None
    return {name:{metric:interval(values) for metric,values in group.items()} for name,group in deltas.items()}


def summarize(run_dirs,output=ROOT/'reports/phase2/comparison'):
    output=Path(output)
    base=np.load(output/'gopt-validation-predictions.npz',allow_pickle=False)
    baseline=json.loads((output/'gopt-validation.json').read_text(encoding='utf-8'))
    records=[]; primary=None
    for path in map(Path,run_dirs):
        cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        metrics=json.loads((path/'best-metrics.json').read_text(encoding='utf-8'))
        with np.load(path/'best-validation-predictions.npz',allow_pickle=False) as pred:
            if not all(np.array_equal(pred[k],base[k]) for k in ('ids','speakers','targets')):
                raise ValueError('Comparison IDs/labels/speakers differ')
            if cfg['seed']==7:
                primary=paired_bootstrap(pred['predictions'],base['predictions'],base['targets'],base['speakers'])
        records.append({'run':path.relative_to(ROOT).as_posix(),'seed':cfg['seed'],'summary':summary,
                        'train':metrics['train'],'validation':metrics['validation'],
                        'mean_baseline':json.loads((path/'mean-baseline.json').read_text(encoding='utf-8')),
                        'resources':json.loads((path/'resources.json').read_text(encoding='utf-8')),
                        'checkpoint_sha256':sha256(path/'best.pt')})
    if len({r['seed'] for r in records})!=len(records): raise ValueError('Repeated seed')
    aggregate={}
    for name in TARGETS:
        aggregate[name]={}
        for metric in ('mse','mae','pcc'):
            values=[r['validation']['targets'][name][metric] for r in records]
            aggregate[name][metric]={'mean':float(np.mean(values)),
                'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,'all_seeds':values,
                'gopt':baseline['targets'][name][metric]}
    result={'scope':'development validation only; not independent generalization test',
            'units':'same four utterance targets in 0–10 units; no phone comparison',
            'comparison_caveat':'GOPT uses canonical phones and multi-granularity objectives; fixed epoch30 selected by validation phone MSE. SSL uses waveform statistics and selects by utterance macro MSE. Not an equal-input/objective ablation.',
            'records':records,'seed_aggregate':aggregate,'gopt':baseline,
            'primary_seed7_minus_gopt_speaker_bootstrap_95pct':primary,
            'uncertainty_note':'1000 paired speaker-cluster resamples; development/model-selection uncertainty remains; not adjusted for multiple targets',
            'test_evaluated':False}
    write_json(output/'summary.json',result)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['target','gopt_mse','ssl_mse_mean','ssl_mse_sd','gopt_pcc','ssl_pcc_mean','ssl_pcc_sd'])
        for name in TARGETS:
            a=aggregate[name]
            w.writerow([name,a['mse']['gopt'],a['mse']['mean'],a['mse']['sample_std'],a['pcc']['gopt'],a['pcc']['mean'],a['pcc']['sample_std']])
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    xx=np.arange(4)
    for ax,metric,label in zip(axes,('mse','pcc'),('Utterance MSE (lower is better)','Utterance PCC (higher is better)')):
        ax.bar(xx-.18,[aggregate[n][metric]['gopt'] for n in TARGETS],.36,label='GOPT Baseline 01')
        ax.bar(xx+.18,[aggregate[n][metric]['mean'] for n in TARGETS],.36,
               yerr=[aggregate[n][metric]['sample_std'] or 0 for n in TARGETS],capsize=3,label='Frozen WavLM (seed mean +/- SD)')
        ax.set_xticks(xx,TARGETS,rotation=15); ax.set_ylabel(label); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same development validation: 500 utterances / 25 speakers')
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=170); plt.close(fig)
    return result
