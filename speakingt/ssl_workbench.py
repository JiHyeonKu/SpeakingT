"""Frozen WavLM utterance assessment, with auditable feature caching and runs.

No official-test access. Four sentence targets in original 0–10 units.
"""
from __future__ import annotations

import csv
import ast
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import threading
import textwrap
import time

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .phase2_data import ROOT, DATA, BASELINE, TARGETS, sha256, write_json, read_audio, preserve_baseline
from .gopt_workbench import metric

ENCODER_REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
ENCODER_DIR = ROOT/'.cache/ssl-models/wavlm-base-plus'/ENCODER_REVISION
FEATURE_VERSION = 'wavlm-base-plus-layers1-12-population-mean-std-fp32-v1'


def extraction_code_hash():
    """Fingerprint executable syntax so comment edits do not invalidate acoustic tensors."""
    syntax=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(FrozenEncoder.extract))),include_attributes=False)
    return hashlib.sha256(syntax.encode()).hexdigest()


class ResourceMonitor:
    """Observed process RSS, process CPU time, wall time; no power estimate."""
    def __enter__(self):
        self.process=psutil.Process()
        self.start=time.perf_counter()
        self.cpu_start=sum(self.process.cpu_times()[:2])
        self.rss_start=self.process.memory_info().rss
        self.peak=self.rss_start
        self.stop=threading.Event()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        def sample():
            while not self.stop.wait(.05):
                self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
        elapsed=time.perf_counter()-self.start
        self.peak=max(self.peak,self.process.memory_info().rss)
        cpu=sum(self.process.cpu_times()[:2])-self.cpu_start
        self.result={'wall_seconds':elapsed,'process_cpu_seconds':cpu,
                     'mean_cpu_cores_used':cpu/elapsed,'observed_peak_rss_bytes':self.peak,
                     'rss_at_start_bytes':self.rss_start,
                     'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                     'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                     'rss_sampling_interval_seconds':.05,'paid_compute_usd':0}


class FrozenEncoder:
    def __init__(self, device='cpu', threads=4, folder=ENCODER_DIR):
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested device unavailable')
        torch.set_num_threads(threads)
        self.device=device
        lock=json.loads((ROOT/'experiments/phase2/encoder-lock.json').read_text(encoding='utf-8'))
        folder=Path(folder)
        from .phase2_data import blob_sha
        for asset in lock['files']:
            p=folder/asset['file']
            digest=sha256(p) if 'sha256' in asset else blob_sha(p.read_bytes())
            if digest != asset.get('sha256',asset.get('git_blob_sha1')):
                raise ValueError('Encoder checksum mismatch')
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(folder,local_files_only=True)
        self.model=WavLMModel.from_pretrained(folder,local_files_only=True,
                                              use_safetensors=False,weights_only=True).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.provenance={'repository':'microsoft/wavlm-base-plus','revision':ENCODER_REVISION,
                         'encoder_lock_sha256':sha256(ROOT/'experiments/phase2/encoder-lock.json'),
                         'feature_version':FEATURE_VERSION,
                         'extract_ast_sha256':extraction_code_hash(),
                         'transformers':importlib.metadata.version('transformers'),
                         'torch':str(torch.__version__),'device':device,'dtype':'float32'}

    def extract(self, samples):
        if samples.ndim!=1 or len(samples)<400 or not np.isfinite(samples).all():
            raise ValueError('Expected finite mono 16kHz waveform')
        # One entire utterance, no padding, truncation, silence removal or augmentation.
        # Pinned Base+ processor has do_normalize=False: no waveform standardization.
        inputs=self.processor(samples,sampling_rate=16000,return_tensors='pt')
        self.model.eval()
        with torch.inference_mode():
            output=self.model(inputs.input_values.to(self.device),output_hidden_states=True)
            hidden=output.hidden_states[1:]  # exactly 12 Transformer layers, no projection layer
            features=torch.stack([torch.cat([h[0].mean(0),h[0].std(0,unbiased=False)]) for h in hidden])
        values=features.cpu().numpy().astype(np.float32)
        if values.shape!=(12,1536) or not np.isfinite(values).all():
            raise ValueError('Unexpected encoder output')
        return values, {'input_samples':len(samples),'input_tensor_shape':list(inputs.input_values.shape),
                        'hidden_tensor_shape':list(output.last_hidden_state.shape),'pooled_shape':list(values.shape)}


def load_manifest(directory=DATA):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if any(r['split'] not in ('train','validation') for r in manifest['rows']):
        raise ValueError('Only development train/validation are allowed')
    ids=[r['id'] for r in manifest['rows']]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate manifest IDs')
    split=json.loads((BASELINE/'split.json').read_text(encoding='utf-8'))
    if manifest['audit']['baseline_split_sha256']!=sha256(BASELINE/'split.json'):
        raise ValueError('Baseline split fingerprint changed')
    for name in ('train','validation'):
        if [r['id'] for r in manifest['rows'] if r['split']==name]!=split[name+'_ids']:
            raise ValueError('Development split changed or official test entered')
    return manifest


def _feature_digest(values):
    return hashlib.sha256(np.asarray(values,dtype='<f4').tobytes()).hexdigest()


def read_feature(path, expected):
    with np.load(path,allow_pickle=False) as archive:
        values=archive['features']
        meta=json.loads(str(archive['metadata']))
    if values.shape!=(12,1536) or not np.isfinite(values).all() or meta['feature_sha256']!=_feature_digest(values):
        raise ValueError('Cached feature content changed')
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError('Cached feature provenance mismatch')
    return values,meta


def extract_features(encoder, rows=None, directory=DATA, on_progress=None):
    directory=Path(directory)
    manifest=load_manifest(directory)
    rows=manifest['rows'] if rows is None else rows
    permitted={r['id']:r for r in manifest['rows']}
    if any(r!=permitted.get(r['id']) for r in rows):
        raise ValueError('Feature request is not from locked development manifest')
    cache=directory/'features'/FEATURE_VERSION
    cache.mkdir(parents=True,exist_ok=True)
    records=[]
    computed=0
    with ResourceMonitor() as resources:
        for i,row in enumerate(rows):
            wav=directory/row['audio_path']
            expected={'id':row['id'],'audio_sha256':sha256(wav),'encoder':encoder.provenance}
            # Re-check raw source identity, even when a feature file exists.
            from .phase2_data import blob_sha
            if blob_sha(wav.read_bytes())!=row['git_blob_sha1']:
                raise ValueError('WAV changed from pinned source')
            path=cache/(row['id']+'.npz')
            if path.exists():
                _,meta=read_feature(path,expected)
            else:
                waveform,pcm_hash=read_audio(wav)
                values,shapes=encoder.extract(waveform)
                meta={**expected,**shapes,'pcm_sha256':pcm_hash,'feature_sha256':_feature_digest(values)}
                partial=path.with_suffix('.partial')
                with partial.open('wb') as f:
                    np.savez(f,features=values,metadata=np.asarray(json.dumps(meta)))
                partial.replace(path)
                computed+=1
            records.append({'id':row['id'],'file':path.relative_to(directory).as_posix(),
                            'sha256':sha256(path),'audio_sha256':expected['audio_sha256']})
            if (i+1)%25==0 or i+1==len(rows):
                seconds=time.perf_counter()-resources.start
                print(f'SSL features {i+1}/{len(rows)}; newly computed={computed}; elapsed={seconds:.1f}s',flush=True)
                if on_progress: on_progress(i+1,len(rows),seconds)
    report={'encoder':encoder.provenance,'dataset_manifest_sha256':sha256(directory/'manifest.json'),
            'count':len(records),'computed':computed,'rows':records,'resources':resources.result}
    name='feature-manifest.json' if len(rows)==2500 else f'feature-preview-{len(rows)}.json'
    manifest_path=directory/name
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ('encoder','dataset_manifest_sha256','rows'):
            if prior[key]!=report[key]: raise ValueError('Existing feature manifest identity differs')
        # Keep provenance hash stable on a cache-only rerun, so saved checkpoints
        # remain reloadable. Timings belong to extraction events, not identity.
    else:
        write_json(manifest_path,report)
    from datetime import datetime
    write_json(directory/'extraction-events'/datetime.now().strftime('%Y%m%d-%H%M%S-%f.json'),report)
    return report


def feature_dataset(directory=DATA):
    directory=Path(directory)
    manifest=load_manifest(directory)
    fm=json.loads((directory/'feature-manifest.json').read_text(encoding='utf-8'))
    if fm['count']!=2500 or fm['dataset_manifest_sha256']!=sha256(directory/'manifest.json'):
        raise ValueError('Incomplete/changed feature dataset')
    index={r['id']:r for r in fm['rows']}
    values=[]
    for row in manifest['rows']:
        stored=index[row['id']]; path=directory/stored['file']
        if sha256(path)!=stored['sha256']:
            raise ValueError('Feature file changed')
        x,_=read_feature(path,{'id':row['id'],'audio_sha256':stored['audio_sha256'],'encoder':fm['encoder']})
        values.append(x)
    return {'features':np.stack(values), 'targets':np.asarray([r['targets'] for r in manifest['rows']],dtype=np.float32),
            'ids':np.asarray([r['id'] for r in manifest['rows']]),
            'speakers':np.asarray([r['speaker'] for r in manifest['rows']]),
            'split':np.asarray([r['split'] for r in manifest['rows']]),
            'feature_manifest_sha256':sha256(directory/'feature-manifest.json'),
            'dataset_manifest_sha256':sha256(directory/'manifest.json'),
            'baseline_split_sha256':manifest['audit']['baseline_split_sha256']}


class AssessmentHead(nn.Module):
    """Learn layer mixing AFTER temporal statistics pooling; not frame attention."""
    def __init__(self, hidden=128, dropout=.2, layers=12, width=1536):
        super().__init__()
        self.layer_logits=nn.Parameter(torch.zeros(layers))
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,4))

    def forward(self, features):
        mixed=(features*self.layer_logits.softmax(0)[None,:,None]).sum(1)
        return self.network(mixed)


@dataclass(frozen=True)
class TrainConfig:
    epochs:int=150
    patience:int=20
    learning_rate:float=1e-3
    weight_decay:float=1e-3
    batch_size:int=64
    hidden:int=128
    dropout:float=.2
    seed:int=7
    threads:int=4
    device:str='cpu'

    def validate(self):
        for k in ('epochs','patience','batch_size','hidden','threads'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError('Invalid '+k)
        if type(self.seed) is not int or not 0<=self.seed<2**32: raise ValueError('Invalid seed')
        if not np.isfinite([self.learning_rate,self.weight_decay,self.dropout]).all(): raise ValueError('Nonfinite config')
        if self.learning_rate<=0 or self.weight_decay<0 or not 0<=self.dropout<1: raise ValueError('Invalid hyperparameter')
        if self.device not in ('cpu','cuda') or (self.device=='cuda' and not torch.cuda.is_available()): raise ValueError('Device unavailable')


def initialize_seed(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)


def score_report(predictions,targets):
    predictions,targets=np.asarray(predictions),np.asarray(targets)
    if predictions.shape!=targets.shape or predictions.ndim!=2 or predictions.shape[1]!=4:
        raise ValueError('Expected N x 4 utterance scores')
    metrics={name:metric(predictions[:,i],targets[:,i]) for i,name in enumerate(TARGETS)}
    correlations=[v['pcc'] for v in metrics.values()]
    return {'unit':'utterance; target scale 0–10; predictions not clipped','targets':metrics,
            'macro_mse':float(np.mean([v['mse'] for v in metrics.values()])),
            'macro_mae':float(np.mean([v['mae'] for v in metrics.values()])),
            'macro_pcc':float(np.mean(correlations)) if all(v is not None for v in correlations) else None}


def normalize_features(features, train_mask):
    values=features[train_mask].astype(np.float64)
    mean=values.mean(0).astype(np.float32)
    std=np.maximum(values.std(0),1e-5).astype(np.float32)
    return ((features-mean)/std).astype(np.float32),mean,std


def predict_features(model, x, device='cpu', batch_size=128):
    model.to(device).eval()
    result=[]
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32)),batch_size=batch_size):
            result.append(model(batch[0].to(device)).cpu().numpy()*10)
    return np.concatenate(result)


def validate_dataset(dataset):
    x,y=dataset['features'],dataset['targets']
    if x.ndim!=3 or x.shape[1:]!=(12,1536) or y.shape!=(len(x),4): raise ValueError('Dataset shapes differ')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y<0)|(y>10)): raise ValueError('Invalid features/labels')
    if len(set(dataset['ids']))!=len(x): raise ValueError('Duplicate utterances')
    tr=dataset['split']=='train'; val=dataset['split']=='validation'
    if not tr.any() or not val.any() or not (tr|val).all(): raise ValueError('Invalid development split')
    if set(dataset['speakers'][tr]) & set(dataset['speakers'][val]): raise ValueError('Speaker overlap')
    return tr,val


def train(dataset,run_dir,config=TrainConfig(),on_epoch=None):
    config.validate()
    tr,val=validate_dataset(dataset)
    run_dir=Path(run_dir)
    if run_dir.exists(): raise ValueError('Run already exists; use a new directory')
    initialize_seed(config)
    x,mean,std=normalize_features(dataset['features'],tr)
    y=dataset['targets']
    model=AssessmentHead(config.hidden,config.dropout).to(config.device)
    with torch.no_grad(): model.network[-1].bias.copy_(torch.tensor(y[tr].mean(0)/10,device=config.device))
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr]/10)),
                      batch_size=config.batch_size,shuffle=True,generator=torch.Generator().manual_seed(config.seed))
    run_dir.mkdir(parents=True)
    provenance={k:dataset.get(k) for k in ('dataset_manifest_sha256','feature_manifest_sha256','baseline_split_sha256')}
    write_json(run_dir/'config.json',{**asdict(config),**provenance,'targets':list(TARGETS),
               'protocol_sha256':sha256(ROOT/'experiments/phase2/protocol.json'),
               'encoder_frozen':True,'feature_version':FEATURE_VERSION,'model_revision':ENCODER_REVISION,
               'selection_metric':'validation macro MSE across four utterance scores, original 0–10 units',
               'feature_normalization':'training utterances only, per-layer per-feature mean/std',
               'source_sha256':sha256(Path(__file__)),
               'versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','transformers','psutil')}})
    write_json(run_dir/'split.json',{s+'_ids':dataset['ids'][dataset['split']==s].tolist() for s in ('train','validation')})
    constant=np.repeat(y[tr].mean(0)[None],int(val.sum()),axis=0)
    write_json(run_dir/'mean-baseline.json',score_report(constant,y[val]))
    best=float('inf'); best_epoch=None; stale=0; history=[]; status='completed'
    with ResourceMonitor() as resources:
        def record(epoch,batch_loss):
            nonlocal best,best_epoch,stale
            pt=predict_features(model,x[tr],config.device)
            pv=predict_features(model,x[val],config.device)
            mt,mv=score_report(pt,y[tr]),score_report(pv,y[val])
            change=sum(float(((v.detach().cpu()-initial[k])**2).sum()) for k,v in model.state_dict().items())**.5
            row={'epoch':epoch,'batch_mse_0_1':batch_loss,'train_mse':mt['macro_mse'],'val_mse':mv['macro_mse'],
                 'train_pcc':mt['macro_pcc'],'val_pcc':mv['macro_pcc'],'weight_change_l2':change,
                 'elapsed_seconds':time.perf_counter()-resources.start}
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':asdict(config),'epoch':epoch,
                   'normalization_mean':torch.from_numpy(mean),'normalization_std':torch.from_numpy(std),
                   'split_ids':{s:dataset['ids'][dataset['split']==s].tolist() for s in ('train','validation')},
                   'provenance':provenance,'feature_version':FEATURE_VERSION,'targets':list(TARGETS)}
            torch.save(state,run_dir/'last.pt')
            if epoch==0: torch.save(state,run_dir/'initial.pt')
            if epoch>0 and mv['macro_mse']<best:
                best=mv['macro_mse']; best_epoch=epoch; stale=0
                torch.save(state,run_dir/'best.pt')
                np.savez(run_dir/'best-validation-predictions.npz',ids=dataset['ids'][val],speakers=dataset['speakers'][val],
                         predictions=pv,targets=y[val])
                write_json(run_dir/'best-metrics.json',{'epoch':epoch,'train':mt,'validation':mv})
            elif epoch>0: stale+=1
            history.append(row)
            with (run_dir/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if epoch==0: writer.writeheader()
                writer.writerow(row)
            write_json(run_dir/f'metrics-{epoch:03d}.json',{'train':mt,'validation':mv})
            print(f"epoch {epoch:3d}: train MSE {mt['macro_mse']:.4f}, validation MSE {mv['macro_mse']:.4f}, PCC {mv['macro_pcc']}",flush=True)
            if on_epoch: on_epoch(history)
        try:
            record(0,None)
            for epoch in range(1,config.epochs+1):
                model.train(); loss_sum=0.; count=0
                for bx,by in loader:
                    bx,by=bx.to(config.device),by.to(config.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss=nn.functional.mse_loss(model(bx),by)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    loss.backward(); optimizer.step()
                    loss_sum+=float(loss.detach())*len(bx); count+=len(bx)
                record(epoch,loss_sum/count)
                if stale>=config.patience:
                    status='early_stopped'; break
        except KeyboardInterrupt:
            status='interrupted'
        except Exception:
            status='failed'; raise
        finally:
            write_json(run_dir/'summary.json',{'status':status,'epochs_completed':history[-1]['epoch'] if history else 0,
                        'best_epoch':best_epoch,'best_validation_macro_mse':best if np.isfinite(best) else None,
                        'test_evaluated':False})
    write_json(run_dir/'resources.json',resources.result)
    return history


def restore(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['feature_version']!=FEATURE_VERSION or saved['targets']!=list(TARGETS): raise ValueError('Incompatible checkpoint')
    model=AssessmentHead(saved['config']['hidden'],saved['config']['dropout'])
    model.load_state_dict(saved['model'],strict=True)
    model.eval()
    return model,saved


def evaluate_checkpoint(path,dataset,split='validation'):
    if split not in ('train','validation'): raise ValueError('Official test is disabled in Phase 2')
    validate_dataset(dataset)
    model,saved=restore(path)
    for key,value in saved['provenance'].items():
        if value!=dataset.get(key): raise ValueError('Checkpoint/dataset provenance mismatch')
    mask=dataset['split']==split
    if saved.get('split_ids') and dataset['ids'][mask].tolist()!=saved['split_ids'][split]:
        raise ValueError('Checkpoint evaluation membership/order differs')
    x=(dataset['features'][mask]-saved['normalization_mean'].numpy())/saved['normalization_std'].numpy()
    predictions=predict_features(model,x)
    return score_report(predictions,dataset['targets'][mask]),predictions


def restore_run_dataset(run_dir,directory=DATA):
    """Rebuild the exact completed run input from checked cache after kernel restart."""
    directory=Path(directory); run_dir=Path(run_dir)
    cfg=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    splits=json.loads((run_dir/'split.json').read_text(encoding='utf-8'))
    manifest=load_manifest(directory)
    lookup={r['id']:r for r in manifest['rows']}
    ids=splits['train_ids']+splits['validation_ids']
    rows=[lookup[i] for i in ids]
    expected_groups=['train']*len(splits['train_ids'])+['validation']*len(splits['validation_ids'])
    if [r['split'] for r in rows]!=expected_groups: raise ValueError('Restored split assignment differs')
    fm_path=directory/'feature-manifest.json'
    fm=json.loads(fm_path.read_text(encoding='utf-8')) if fm_path.exists() else None
    cache_index={r['id']:r for r in fm['rows']} if fm else {}
    features=[]
    for row in rows:
        wav=directory/row['audio_path']
        path=directory/'features'/FEATURE_VERSION/(row['id']+'.npz')
        expected={'id':row['id'],'audio_sha256':sha256(wav)}
        if fm is not None:
            if sha256(path)!=cache_index[row['id']]['sha256']: raise ValueError('Restored feature file changed')
            expected['encoder']=fm['encoder']
        values,meta=read_feature(path,expected)
        if meta['encoder']['revision']!=ENCODER_REVISION or meta['encoder']['feature_version']!=FEATURE_VERSION:
            raise ValueError('Restored encoder differs')
        features.append(values)
    fingerprint_paths={'dataset_manifest_sha256':directory/'manifest.json',
                       'feature_manifest_sha256':directory/'feature-manifest.json',
                       'baseline_split_sha256':BASELINE/'split.json'}
    for key,path in fingerprint_paths.items():
        if cfg.get(key) is not None and cfg[key]!=sha256(path): raise ValueError('Restored source hash differs')
    return {'features':np.stack(features),'targets':np.asarray([r['targets'] for r in rows],dtype=np.float32),
            'ids':np.asarray(ids),'speakers':np.asarray([r['speaker'] for r in rows]),
            'split':np.asarray(expected_groups),**{key:cfg.get(key) for key in fingerprint_paths}}


def plot_history(history):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    epochs=[int(r['epoch']) for r in history]
    for prefix,label in (('train','Train'),('val','Validation')):
        axes[0].plot(epochs,[float(r[prefix+'_mse']) for r in history],label=label)
        axes[1].plot(epochs,[np.nan if r[prefix+'_pcc'] in (None,'') else float(r[prefix+'_pcc']) for r in history],label=label)
    axes[0].set(xlabel='Epoch',ylabel='Macro utterance MSE (0–10 scale)')
    axes[1].set(xlabel='Epoch',ylabel='Macro utterance PCC',ylim=(-1,1))
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    return fig


def gopt_comparison(dataset,output):
    """Exact Baseline 01 checkpoint, exact development validation IDs, four sentence targets."""
    from . import gopt_workbench as gopt
    preserve_baseline()
    base=gopt.load_data(ROOT/'data/gopt','train')
    lookup={uid:i for i,uid in enumerate(base['ids'])}
    val=dataset['split']=='validation'
    ids=dataset['ids'][val]
    expected=json.loads((BASELINE/'split.json').read_text(encoding='utf-8'))['validation_ids']
    if ids.tolist()!=expected: raise ValueError('Not the same ordered Baseline 01 validation')
    selected=gopt.subset(base,np.array([lookup[uid] for uid in ids]))
    model,norm=gopt.load_checkpoint(BASELINE/'best.pt')
    _,prediction=gopt.evaluate(model,selected,norm,return_predictions=True)
    p=prediction['utterance'][:,[0,2,3,4]]
    if not np.array_equal(selected['label_utt'][:,[0,2,3,4]],dataset['targets'][val]):
        raise ValueError('Baseline/SSL target mismatch')
    result=score_report(p,dataset['targets'][val])
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/'gopt-validation.json',result)
    np.savez(output/'gopt-validation-predictions.npz',ids=ids,speakers=dataset['speakers'][val],
             predictions=p,targets=dataset['targets'][val])
    return result
