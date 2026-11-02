"""Phase 3: raw-waveform WavLM partial fine-tuning. No feature-cache training."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .phase2_data import ROOT, DATA, BASELINE, TARGETS, sha256, blob_sha, read_audio, write_json
from .ssl_workbench import (ENCODER_DIR, ENCODER_REVISION, FEATURE_VERSION, AssessmentHead,
                           load_manifest, score_report, initialize_seed, ResourceMonitor, plot_history)

SCHEMA=1
WARM_RUN='runs/ssl/notebook-full-seed7-20261102-103405-853366'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@dataclass(frozen=True)
class Config:
    top_layers:int=2
    hidden:int=128
    dropout:float=.5
    encoder_lr:float=1e-5
    head_lr:float=1e-4
    weight_decay:float=.01
    batch_size:int=2
    accumulation:int=2
    clip_norm:float=1.
    epochs:int=1
    patience:int=3
    seed:int=7
    threads:int=4
    device:str='cpu'
    precision:str='fp32'
    stage:str='smoke'
    allow_full:bool=False
    warm_start:str|None=WARM_RUN

    def validate(self):
        if self.top_layers not in (0,1,2): raise ValueError('top_layers must be 0, 1, or 2')
        for k in ('hidden','batch_size','accumulation','epochs','patience','threads'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError('Invalid '+k)
        if not 0<=self.seed<2**32: raise ValueError('Invalid seed')
        if not np.isfinite([self.dropout,self.encoder_lr,self.head_lr,self.weight_decay,self.clip_norm]).all():
            raise ValueError('Nonfinite configuration')
        if not 0<=self.dropout<1 or min(self.encoder_lr,self.head_lr,self.clip_norm)<=0 or self.weight_decay<0:
            raise ValueError('Invalid training configuration')
        if self.device not in ('cpu','cuda'): raise ValueError('Only cpu/cuda supported')
        if self.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        if self.precision not in ('fp32','fp16','bf16'): raise ValueError('Unknown precision')
        if self.device=='cpu' and self.precision!='fp32': raise ValueError('CPU verification uses fp32')
        if self.precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('BF16 unsupported')
        if self.stage not in ('smoke','pilot','full'): raise ValueError('Unknown stage')
        if self.stage=='full' and (self.device!='cuda' or not self.allow_full):
            raise ValueError('Full training requires CUDA and explicit allow_full=True')
        if self.stage=='smoke' and self.epochs>2: raise ValueError('Smoke verification is limited to 2 epochs')
        if self.stage=='pilot' and (self.device!='cuda' or self.epochs>2):
            raise ValueError('GPU pilot requires CUDA and at most 2 epochs')


def environment():
    import platform
    import psutil
    return {'python':platform.python_version(),'os':platform.platform(),
            'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','psutil')},
            'ram_bytes':psutil.virtual_memory().total,'cuda':torch.cuda.is_available(),
            'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0}


def data_identity():
    return {'manifest_sha256':sha256(DATA/'manifest.json'), 'split_sha256':sha256(BASELINE/'split.json'),
            'encoder_lock_sha256':sha256(ROOT/'experiments/phase2/encoder-lock.json'),
            'revision':ENCODER_REVISION,'targets':list(TARGETS)}


def select_rows(stage='smoke'):
    rows=load_manifest()['rows']
    if stage=='full': return rows
    if stage not in ('smoke','pilot'): raise ValueError('Unknown stage')
    # Speaker-diverse deterministic coverage, including the longest utterance in a GPU pilot.
    result=[]
    for group,count in [('train',16 if stage=='smoke' else 48),('validation',4 if stage=='smoke' else 16)]:
        candidates=[r for r in rows if r['split']==group]
        if stage=='pilot':
            longest=max(candidates,key=lambda r:r['bytes']); result.append(longest); count-=1
        chosen=[]; seen=set()
        for r in candidates:
            if r['speaker'] not in seen and r not in result:
                chosen.append(r); seen.add(r['speaker'])
                if len(chosen)==count: break
        result.extend(chosen)
    return result


def validate_rows(rows,cfg):
    cfg.validate()
    canonical={r['id']:r for r in load_manifest()['rows']}
    if len({r['id'] for r in rows})!=len(rows) or any(canonical.get(r['id'])!=r for r in rows):
        raise ValueError('Duplicate, changed, or non-development rows')
    groups={g:[r for r in rows if r['split']==g] for g in ('train','validation')}
    if any(not v for v in groups.values()): raise ValueError('Both development groups required')
    if {r['speaker'] for r in groups['train']} & {r['speaker'] for r in groups['validation']}:
        raise ValueError('Speaker leakage')
    if cfg.stage=='full' and rows!=load_manifest()['rows']: raise ValueError('Full split differs')
    if cfg.stage=='smoke' and len(rows)>20: raise ValueError('CPU smoke limited to 20 WAVs')
    if cfg.stage=='pilot' and len(rows)>64: raise ValueError('Pilot limited to 64 WAVs')
    return groups


class WaveDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; path=DATA/row['audio_path']
        if blob_sha(path.read_bytes())!=row['git_blob_sha1']: raise ValueError('Raw WAV provenance mismatch')
        samples,_=read_audio(path)
        return torch.from_numpy(samples),torch.tensor(row['targets'],dtype=torch.float32),row['id']


def collate(items):
    wave,targets,ids=zip(*items)
    lengths=torch.tensor([len(w) for w in wave])
    padded=pad_sequence(wave,batch_first=True)
    mask=torch.arange(padded.shape[1])[None,:]<lengths[:,None]
    return padded,mask,torch.stack(targets),list(ids)


def statistics(hidden,mask):
    """FP32, valid frames only, population variance; zero variance has finite gradient."""
    x=hidden.float(); w=mask.unsqueeze(-1).to(x.dtype)
    count=w.sum(1).clamp_min(1)
    mean=(x*w).sum(1)/count
    variance=((x-mean[:,None,:]).square()*w).sum(1)/count
    std=variance.clamp_min(1e-12).sqrt()
    return torch.cat([mean,std],-1)


class PartialWavLM(nn.Module):
    def __init__(self,cfg:Config,backbone=None):
        super().__init__(); self.cfg=cfg
        if backbone is None:
            from transformers import WavLMModel, Wav2Vec2FeatureExtractor
            if importlib.metadata.version('transformers')!='4.57.6':
                raise ValueError('Use audited transformers==4.57.6')
            for asset in read_json(ROOT/'experiments/phase2/encoder-lock.json')['files']:
                path=ENCODER_DIR/asset['file']
                digest=sha256(path) if 'sha256' in asset else blob_sha(path.read_bytes())
                if digest!=asset.get('sha256',asset.get('git_blob_sha1')): raise ValueError('Encoder asset changed')
            processor=Wav2Vec2FeatureExtractor.from_pretrained(ENCODER_DIR,local_files_only=True)
            if processor.do_normalize or processor.sampling_rate!=16000: raise ValueError('Processor settings changed')
            backbone,info=WavLMModel.from_pretrained(ENCODER_DIR,local_files_only=True,use_safetensors=False,
                                                    weights_only=True,output_loading_info=True)
            if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise ValueError(f'Pretrained load was not exact: {info}')
        self.wavlm=backbone
        if len(backbone.encoder.layers)!=12 or backbone.config.add_adapter: raise ValueError('Expected 12-layer base encoder')
        self.wavlm.requires_grad_(False)
        if cfg.top_layers:
            for layer in self.wavlm.encoder.layers[-cfg.top_layers:]: layer.requires_grad_(True)
        width=backbone.config.hidden_size*2
        self.head=AssessmentHead(cfg.hidden,cfg.dropout,width=width)
        self.register_buffer('mean',torch.zeros(12,width))
        self.register_buffer('std',torch.ones(12,width))
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        # No LayerDrop/SpecAugment or frozen-block dropout. Gradients are independent of eval mode.
        self.wavlm.eval()
        if self.cfg.top_layers:
            for layer in self.wavlm.encoder.layers[-self.cfg.top_layers:]: layer.train(mode)
        self.head.train(mode)
        return self

    def features(self,wave,mask):
        if wave.ndim!=2 or mask.shape!=wave.shape or mask.dtype!=torch.bool or not torch.isfinite(wave).all():
            raise ValueError('Expected finite padded waveforms and boolean sample mask')
        lengths=mask.sum(1)
        if (lengths<400).any() or not torch.equal(mask,torch.arange(wave.shape[1],device=wave.device)[None,:]<lengths[:,None]):
            raise ValueError('Expected nonempty right-padded audio')
        # GroupNorm sees only the original waveform, never padded samples.
        projected=[]
        for w,n in zip(wave,lengths.tolist()):
            conv=self.wavlm.feature_extractor(w[None,:n]).transpose(1,2)
            projected.append(self.wavlm.feature_projection(conv)[0][0])
        sizes=torch.tensor([len(x) for x in projected],device=wave.device)
        padded=pad_sequence(projected,batch_first=True)
        valid=torch.arange(padded.shape[1],device=wave.device)[None,:]<sizes[:,None]
        expected=self.wavlm._get_feat_extract_output_lengths(lengths)
        if not torch.equal(sizes,expected): raise ValueError('Convolution frame lengths disagree')
        output=self.wavlm.encoder(padded,attention_mask=valid,output_hidden_states=True,return_dict=True)
        pooled=torch.stack([statistics(h,valid) for h in output.hidden_states[1:]],dim=1)
        return pooled,valid

    def forward(self,wave,mask):
        features,_=self.features(wave,mask)
        return self.head((features-self.mean)/self.std)


def param_digest(model,frozen=True):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if (not p.requires_grad)==frozen:
            digest.update(name.encode()); digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grad_report(model):
    result={}
    for label,params in [('encoder',[p for p in model.wavlm.parameters() if p.requires_grad]),
                         ('head',list(model.head.parameters()))]:
        grads=[p.grad for p in params if p.grad is not None]
        result[label]={'parameters':sum(p.numel() for p in params),'with_gradient':len(grads),
                       'norm':sum(float(g.detach().float().square().sum()) for g in grads)**.5,
                       'finite':all(bool(torch.isfinite(g).all()) for g in grads)}
    result['frozen_gradient_tensors']=sum(p.grad is not None for p in model.parameters() if not p.requires_grad)
    return result


def autocast(cfg):
    return (torch.autocast('cuda',dtype=torch.float16 if cfg.precision=='fp16' else torch.bfloat16)
            if cfg.precision!='fp32' else nullcontext())


def initialize_model(cfg,rows):
    initialize_seed(cfg); model=PartialWavLM(cfg).to(cfg.device)
    provenance={'initialization':'fresh_head','normalization':'selected training WAVs only'}
    if cfg.warm_start:
        path=ROOT/cfg.warm_start
        source_cfg=read_json(path/'config.json')
        saved=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        expected=load_manifest()
        if source_cfg['hidden']!=cfg.hidden or source_cfg['dropout']!=cfg.dropout:
            raise ValueError('Warm-start head settings differ')
        expected_split={s:[r['id'] for r in expected['rows'] if r['split']==s] for s in ('train','validation')}
        required={'dataset_manifest_sha256':sha256(DATA/'manifest.json'),
                  'feature_manifest_sha256':sha256(DATA/'feature-manifest.json'),
                  'baseline_split_sha256':sha256(BASELINE/'split.json')}
        if saved.get('split_ids')!=expected_split or saved['provenance']!=required or saved['feature_version']!=FEATURE_VERSION:
            raise ValueError('Warm-start provenance/split differs')
        if source_cfg['model_revision']!=ENCODER_REVISION or saved['targets']!=list(TARGETS):
            raise ValueError('Warm-start encoder/targets differ')
        model.head.load_state_dict(saved['model'],strict=True)
        model.mean.copy_(saved['normalization_mean']); model.std.copy_(saved['normalization_std'])
        provenance={'initialization':'phase2_head_warm_start','source_run':cfg.warm_start,
                    'source_checkpoint_sha256':sha256(path/'best.pt'),'source_epoch':saved['epoch'],
                    'normalization':'fixed Phase 2 statistics fitted on the original 2000 training utterances'}
    else:
        model.eval(); values=[]
        with torch.no_grad():
            for batch in DataLoader(WaveDataset([r for r in rows if r['split']=='train']),batch_size=cfg.batch_size,collate_fn=collate):
                values.append(model.features(batch[0].to(cfg.device),batch[1].to(cfg.device))[0].cpu())
        features=torch.cat(values).double()
        model.mean.copy_(features.mean(0).float()); model.std.copy_(features.std(0,unbiased=False).float().clamp_min(1e-5))
        labels=torch.tensor([r['targets'] for r in rows if r['split']=='train'])
        with torch.no_grad(): model.head.network[-1].bias.copy_(labels.mean(0).to(cfg.device)/10)
    if not torch.isfinite(model.mean).all() or not torch.isfinite(model.std).all() or (model.std<=0).any():
        raise ValueError('Invalid normalization')
    return model,provenance


def evaluate(model,rows,batch_size=2):
    model.eval(); predictions=[]; targets=[]; ids=[]
    device=next(model.parameters()).device
    with torch.no_grad():
        for wave,mask,y,names in DataLoader(WaveDataset(rows),batch_size=batch_size,collate_fn=collate):
            p=model(wave.to(device),mask.to(device)).float().cpu().numpy()*10
            if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
            predictions.append(p); targets.append(y.numpy()); ids.extend(names)
    p=np.concatenate(predictions); y=np.concatenate(targets)
    return score_report(p,y),p,np.asarray(ids)


def make_optimizer(model,cfg):
    groups=[{'params':list(model.head.parameters()),'lr':cfg.head_lr,'name':'head'}]
    encoder=[p for p in model.wavlm.parameters() if p.requires_grad]
    if encoder: groups.append({'params':encoder,'lr':cfg.encoder_lr,'name':'encoder'})
    return torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)


def train_epoch(model,rows,cfg,optimizer,scaler,epoch):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loader=DataLoader(WaveDataset(rows),batch_size=cfg.batch_size,shuffle=True,collate_fn=collate,
                      generator=torch.Generator().manual_seed(cfg.seed+epoch))
    batch_count=len(loader); total_loss=0.; seen=0; report=None; steps=0
    # Sum per-example losses then divide by the actual accumulated example count, including the tail.
    group_examples=0
    for i,(wave,mask,y,_) in enumerate(loader):
        with autocast(cfg):
            prediction=model(wave.to(cfg.device),mask.to(cfg.device))
            loss=nn.functional.mse_loss(prediction.float(),y.to(cfg.device)/10)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        scaler.scale(loss*len(y)).backward()
        group_examples+=len(y); total_loss+=float(loss.detach())*len(y); seen+=len(y)
        if (i+1)%cfg.accumulation==0 or i+1==batch_count:
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(group_examples)
            current=grad_report(model)
            if current['frozen_gradient_tensors'] or any(not current[k]['finite'] for k in ('encoder','head')):
                raise ValueError('Invalid/frozen gradient; lower LR or use fp32')
            if cfg.top_layers and current['encoder']['norm']==0: raise ValueError('Encoder gradient missing')
            if report is None: report=current
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg.clip_norm,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            group_examples=0; steps+=1
    return total_loss/seen,report,steps


def atomic_checkpoint(path,state):
    path=Path(path); temporary=path.with_suffix('.partial')
    torch.save(state,temporary); temporary.replace(path)
    write_json(path.with_suffix('.sha256.json'),{'sha256':sha256(path)})


def restore(checkpoint,device='cpu'):
    checkpoint=Path(checkpoint)
    if sha256(checkpoint)!=read_json(checkpoint.with_suffix('.sha256.json'))['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['schema']!=SCHEMA or saved['identity']!=data_identity(): raise ValueError('Checkpoint provenance mismatch')
    if saved['source_sha256']!=sha256(Path(__file__)): raise ValueError('Training code differs; inspect before migration')
    if saved['environment']['versions']['transformers']!=importlib.metadata.version('transformers'):
        raise ValueError('Transformers version mismatch')
    cfg=Config(**saved['config']); torch.set_num_threads(cfg.threads); model=PartialWavLM(cfg)
    model.load_state_dict(saved['model'],strict=True)
    if [n for n,p in model.named_parameters() if p.requires_grad]!=saved['trainable_names']:
        raise ValueError('Trainable layer set changed')
    canonical={r['id']:r for r in load_manifest()['rows']}
    rows=[canonical[uid] for uid in saved['row_ids']]
    if [r['split'] for r in rows]!=saved['row_splits']: raise ValueError('Split assignment differs')
    model.to(device).eval()
    return model,saved,rows


def run(cfg=Config(),run_dir=None,resume=None,on_epoch=None):
    cfg.validate(); rows=select_rows(cfg.stage); groups=validate_rows(rows,cfg)
    initialize_seed(cfg)
    directory=Path(run_dir) if run_dir else ROOT/'runs/phase3'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if directory.exists(): raise ValueError('Run already exists; use a new run directory')
    directory.mkdir(parents=True)
    history=[]; best=float('inf'); best_epoch=None; stale=0; first_epoch=1; status='running'; failure=None
    with ResourceMonitor() as monitor:
        try:
            if resume:
                model,saved,restored_rows=restore(resume,cfg.device)
                allowed={'epochs','allow_full'}
                if any(v!=saved['config'][k] for k,v in asdict(cfg).items() if k not in allowed) or restored_rows!=rows:
                    raise ValueError('Resume configuration/split differs')
                provenance={**saved['initialization'],'resume_checkpoint_sha256':sha256(resume)}
            else:
                model,provenance=initialize_model(cfg,rows)
            optimizer=make_optimizer(model,cfg)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2)
            scaler=torch.amp.GradScaler('cuda',enabled=cfg.device=='cuda' and cfg.precision=='fp16')
            if resume:
                optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
                torch.set_rng_state(saved['rng_cpu']); random.setstate(saved['rng_python'])
                if cfg.device=='cuda': torch.cuda.set_rng_state_all(saved['rng_cuda'])
                first_epoch=saved['epoch']+1; best=saved['best']; best_epoch=saved['best_epoch']; stale=saved['stale']
                if first_epoch>cfg.epochs or stale>=cfg.patience: raise ValueError('Checkpoint already completed requested training/early stopping')
                # Preserve the inherited best in the new run, without changing the parent run.
                import shutil
                parent_best=Path(resume).parent/'best.pt'
                parent=torch.load(parent_best,map_location='cpu',weights_only=True)
                if parent['epoch']!=best_epoch or parent['identity']!=saved['identity'] or parent['row_ids']!=saved['row_ids']:
                    raise ValueError('Parent best checkpoint does not match resume state')
                shutil.copyfile(parent_best,directory/'best.pt')
                for name in ('best.sha256.json','best-metrics.json','best-validation-predictions.npz'):
                    shutil.copyfile(Path(resume).parent/name,directory/name)
                del parent
            frozen_before=param_digest(model)
            encoder_before={n:p.detach().cpu().clone() for n,p in model.wavlm.named_parameters() if p.requires_grad}
            write_json(directory/'config.json',{'config':asdict(cfg),'initialization':provenance,'identity':data_identity(),
                       'environment':environment(),'source_sha256':sha256(Path(__file__)),
                       'normalization_policy':'fixed training-only statistics; not re-estimated while fine-tuning',
                       'selection':'validation macro MSE; epoch 0 eligible; development only',
                       'encoder_policy':'frozen modules eval; selected top layers train; no SpecAugment/LayerDrop'})
            write_json(directory/'split.json',{g:[r['id'] for r in v] for g,v in groups.items()})
            def record(epoch,batch_loss=None,gradients=None,steps=0):
                nonlocal best,best_epoch,stale
                mt,_,_=evaluate(model,groups['train'],cfg.batch_size)
                mv,p,ids=evaluate(model,groups['validation'],cfg.batch_size)
                scheduler.step(mv['macro_mse']) if epoch>0 else None
                improved=mv['macro_mse']<best
                if improved: best=mv['macro_mse']; best_epoch=epoch; stale=0
                elif epoch>0: stale+=1
                change=sum(float((p.detach().cpu()-encoder_before[n]).square().sum()) for n,p in model.wavlm.named_parameters() if p.requires_grad)**.5
                row={'epoch':epoch,'batch_mse_0_1':batch_loss,'train_mse':mt['macro_mse'],'val_mse':mv['macro_mse'],
                     'train_pcc':mt['macro_pcc'],'val_pcc':mv['macro_pcc'],'encoder_delta_l2':change,
                     'encoder_grad_norm':None if gradients is None else gradients['encoder']['norm'],
                     'optimizer_steps':steps,'head_lr':optimizer.param_groups[0]['lr'],
                     'encoder_lr':optimizer.param_groups[-1]['lr'] if cfg.top_layers else 0,
                     'elapsed_seconds':time.perf_counter()-monitor.start}
                frozen_after=param_digest(model)
                if frozen_before!=frozen_after: raise ValueError('Frozen parameters changed')
                state={'schema':SCHEMA,'model':model.state_dict(),'config':asdict(cfg),'epoch':epoch,'best':best,
                       'best_epoch':best_epoch,'stale':stale,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                       'scaler':scaler.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_python':random.getstate(),
                       'rng_cuda':torch.cuda.get_rng_state_all() if cfg.device=='cuda' else [],
                       'identity':data_identity(),'initialization':provenance,'environment':environment(),
                       'source_sha256':sha256(Path(__file__)),'row_ids':[r['id'] for r in rows],
                       'row_splits':[r['split'] for r in rows],
                       'trainable_names':[n for n,v in model.named_parameters() if v.requires_grad]}
                atomic_checkpoint(directory/'last.pt',state)
                np.savez(directory/'last-validation-predictions.npz',ids=ids,predictions=p,
                         targets=np.asarray([r['targets'] for r in groups['validation']]))
                if improved:
                    atomic_checkpoint(directory/'best.pt',state)
                    write_json(directory/'best-metrics.json',{'epoch':epoch,'train':mt,'validation':mv})
                    np.savez(directory/'best-validation-predictions.npz',ids=ids,predictions=p,
                             targets=np.asarray([r['targets'] for r in groups['validation']]),
                             speakers=np.asarray([r['speaker'] for r in groups['validation']]))
                history.append(row)
                with (directory/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(row))
                    if len(history)==1:w.writeheader()
                    w.writerow(row)
                write_json(directory/f'metrics-{epoch:03d}.json',{'train':mt,'validation':mv,'gradients':gradients,
                           'frozen_sha256':frozen_after,'encoder_delta_l2':change})
                print(f"epoch {epoch}: Train MSE={mt['macro_mse']:.5f}, Validation MSE={mv['macro_mse']:.5f}, encoder delta={change:.6f}",flush=True)
                if on_epoch:on_epoch(history)
            if not resume: record(0)
            for epoch in range(first_epoch,cfg.epochs+1):
                loss,gradients,steps=train_epoch(model,groups['train'],cfg,optimizer,scaler,epoch)
                record(epoch,loss,gradients,steps)
                if stale>=cfg.patience: status='early_stopped'; break
            else: status='completed'
        except KeyboardInterrupt:
            status='interrupted'
        except torch.OutOfMemoryError as error:
            status='oom'; failure=error
        except Exception as error:
            status='failed'; failure=error
        finally:
            write_json(directory/'summary.json',{'status':status,'best_epoch':best_epoch,
                       'best_validation_macro_mse':best if np.isfinite(best) else None,
                       'last_recorded_epoch':history[-1]['epoch'] if history else None,
                       'test_evaluated':False,'scope':'small correctness check' if cfg.stage=='smoke' else 'development validation'})
    resources=dict(monitor.result); resources.pop('paid_compute_usd',None)
    resources['billing_measured']=False
    if cfg.device=='cpu':resources['paid_gpu_compute_usd']=0
    write_json(directory/'resources.json',resources)
    if failure is not None: raise failure
    if history:
        import matplotlib.pyplot as plt
        fig=plot_history(history); fig.savefig(directory/'learning-curves.png',dpi=160); plt.close(fig)
    return directory
