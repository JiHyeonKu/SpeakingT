"""Run the same Phase 2 operations as the notebook, with bounded stage selection."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from speakingt import ssl_workbench as ssl
from speakingt.phase2_data import ROOT, DATA, write_json


def smoke(encoder):
    manifest=ssl.load_manifest()
    rows=[r for r in manifest['rows'] if r['split']=='train'][:16]+[r for r in manifest['rows'] if r['split']=='validation'][:4]
    report=ssl.extract_features(encoder,rows)
    values=[]
    for row,item in zip(rows,report['rows']):
        x,_=ssl.read_feature(DATA/item['file'],{'id':row['id']})
        values.append(x)
    dataset={'features':np.stack(values),'targets':np.array([r['targets'] for r in rows],dtype=np.float32),
             'ids':np.array([r['id'] for r in rows]),'speakers':np.array([r['speaker'] for r in rows]),
             'split':np.array([r['split'] for r in rows])}
    run=ROOT/'runs/ssl'/datetime.now().strftime('stage-b-%Y%m%d-%H%M%S-%f')
    history=ssl.train(dataset,run,ssl.TrainConfig(epochs=3,patience=3,batch_size=8))
    a,p1=ssl.evaluate_checkpoint(run/'best.pt',dataset)
    _,p2=ssl.evaluate_checkpoint(run/'best.pt',dataset)
    assert np.array_equal(p1,p2)
    assert history[-1]['weight_change_l2']>0
    assert all(not p.requires_grad and p.grad is None for p in encoder.model.parameters())
    out={'stage':'B','real_wavs':20,'train':16,'validation':4,'run':run.relative_to(ROOT).as_posix(),
         'weight_change_l2':history[-1]['weight_change_l2'],'reloaded_predictions_identical':True,
         'encoder_all_frozen_no_gradients':True,'feature_shape':list(dataset['features'].shape),
         'not_performance_evidence':True,'resources':report['resources']}
    write_json(ROOT/'reports/phase2/stage-b.json',out)
    print(json.dumps(out,indent=2),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','features','train'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,42])
    args=parser.parse_args()
    if args.stage in ('smoke','features'):
        with ssl.ResourceMonitor() as resources:
            encoder=ssl.FrozenEncoder()
        write_json(ROOT/'reports/phase2/stage-a.json',{'model_parameters':encoder.parameter_count,
                   'trainable_encoder_parameters':sum(p.numel() for p in encoder.model.parameters() if p.requires_grad),
                   'device':encoder.device,'resources':resources.result,'provenance':encoder.provenance})
        if args.stage=='smoke': smoke(encoder)
        else: ssl.extract_features(encoder)
    else:
        dataset=ssl.feature_dataset()
        comparison=ROOT/'reports/phase2/comparison'
        ssl.gopt_comparison(dataset,comparison)
        runs=[]
        for seed in args.seeds:
            run=ROOT/'runs/ssl'/datetime.now().strftime(f'stage-c-seed{seed}-%Y%m%d-%H%M%S-%f')
            history=ssl.train(dataset,run,ssl.TrainConfig(seed=seed))
            fig=ssl.plot_history(history); fig.savefig(run/'learning-curves.png',dpi=160)
            import matplotlib.pyplot as plt
            plt.close(fig)
            metrics,_=ssl.evaluate_checkpoint(run/'best.pt',dataset)
            write_json(run/'restored-validation.json',metrics)
            runs.append(run.relative_to(ROOT).as_posix())
        write_json(ROOT/'reports/phase2/comparison/runs.json',{'runs':runs,'seeds':args.seeds,'test_evaluated':False})


if __name__=='__main__': main()
