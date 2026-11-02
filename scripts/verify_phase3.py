"""Real pinned-model correctness probes; writes only Phase 3 artifacts."""
from dataclasses import replace
import gc
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt import partial_wavlm as p3
from speakingt.phase2_data import ROOT, write_json

def probe():
    cfg=p3.Config(); rows=p3.select_rows()
    with p3.ResourceMonitor() as resources:
        model,source=p3.initialize_model(cfg,rows)
        items=[p3.WaveDataset(rows)[i] for i in (0,1)]
        wave,mask,y,_=p3.collate(items)
        model.eval()
        with torch.no_grad():
            pooled,valid=model.features(wave,mask)
            single=[]; native=[]
            for item in items:
                w=item[0][None]
                single.append(model.features(w,torch.ones_like(w,dtype=torch.bool))[0])
                out=model.wavlm(w,output_hidden_states=True).hidden_states[1:]
                native.append(torch.stack([torch.cat([h[0].mean(0),h[0].std(0,unbiased=False)]) for h in out])[None])
            single=torch.cat(single); native=torch.cat(native)
            batch_error=float((pooled-single).abs().max())
            native_error=float((single-native).abs().max())
            torch.testing.assert_close(pooled,single,rtol=2e-4,atol=2e-4)
            torch.testing.assert_close(single,native,rtol=2e-5,atol=2e-5)
            altered=wave.clone(); altered[~mask]=1000
            torch.testing.assert_close(model.features(altered,mask)[0],pooled,rtol=0,atol=0)
            saved=torch.load(ROOT/cfg.warm_start/'best.pt',map_location='cpu',weights_only=True)
            old_head=p3.AssessmentHead(128,.5); old_head.load_state_dict(saved['model']); old_head.eval()
            expected=old_head((native-saved['normalization_mean'])/saved['normalization_std'])*10
            torch.testing.assert_close(model(wave,mask)*10,expected,rtol=2e-4,atol=2e-4)
        frozen=p3.param_digest(model)
        before={n:v.detach().clone() for n,v in model.wavlm.named_parameters() if v.requires_grad}
        optimizer=p3.make_optimizer(model,cfg); model.train()
        loss=torch.nn.functional.mse_loss(model(wave,mask),y/10); loss.backward()
        gradients=p3.grad_report(model)
        assert gradients['encoder']['norm']>0 and gradients['encoder']['finite']
        assert gradients['frozen_gradient_tensors']==0
        optimizer.step()
        changes={str(i):sum(float((v.detach()-before[n]).square().sum()) for n,v in model.wavlm.named_parameters()
                            if n.startswith(f'encoder.layers.{i}.'))**.5 for i in (10,11)}
        assert all(v>0 for v in changes.values()) and p3.param_digest(model)==frozen
        result={'real_wavs':2,'native_forward_max_abs_error':native_error,'batch_vs_single_pooling_max_abs_error':batch_error,
                'padding_values_ignored':True,'warm_start_matches_phase2_head':True,
                'feature_shape':list(pooled.shape),'valid_frames':valid.sum(1).tolist(),
                'gradient':gradients,'per_layer_update_l2':changes,'frozen_sha256_unchanged':True,
                'initialization':source,'environment':p3.environment()}
    result['resources']=resources.result
    write_json(ROOT/'experiments/phase3/stage-a.json',result)
    print(result,flush=True)

if __name__=='__main__': probe()
