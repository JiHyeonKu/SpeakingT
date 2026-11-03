"""Compare actual uninterrupted/resumed checkpoints, including optimizer and RNG state."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt.phase2_data import ROOT,sha256,write_json
from speakingt.partial_wavlm import read_json


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:equal(a[key],b[key],f'{path}.{key}')
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}[{i}]')
    else: assert a==b,path


def verify(continuous,resumed,output):
    paths=[ROOT/continuous/'last.pt',ROOT/resumed/'last.pt']
    for path in paths:
        assert sha256(path)==read_json(path.with_suffix('.sha256.json'))['sha256']
    states=[torch.load(path,map_location='cpu',weights_only=True) for path in paths]
    fields=('model','optimizer','scheduler','scaler','rng_cpu','rng_python','rng_cuda',
            'epoch','best','best_epoch','stale','identity','row_ids','row_splits','trainable_names','source_sha256')
    for key in fields:equal(states[0][key],states[1][key],key)
    with np.load(paths[0].parent/'last-validation-predictions.npz') as a, \
         np.load(paths[1].parent/'last-validation-predictions.npz') as b:
        for key in ('ids','predictions','targets'):np.testing.assert_array_equal(a[key],b[key])
        prediction_count=len(a['ids'])
    result={'continuous_run':continuous,'resumed_run':resumed,'epoch':states[0]['epoch'],
            'all_checked_fields_bitwise_equal':True,'checked_fields':list(fields),
            'model_tensors_checked':len(states[0]['model']),'validation_predictions':prediction_count,
            'validation_predictions_bitwise_equal':True,'checkpoint_sha256':[sha256(p) for p in paths],
            'scope':'CPU smoke correctness; not a performance experiment'}
    write_json(ROOT/output,result);print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continuous',required=True)
    parser.add_argument('--resumed',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();verify(args.continuous,args.resumed,args.output)
