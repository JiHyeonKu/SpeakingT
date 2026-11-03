"""Small random backbones test correctness, not pretrained model performance."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

HAS=all(importlib.util.find_spec(x) is not None for x in ('torch','transformers','psutil'))

@unittest.skipUnless(HAS,'optional speech research dependencies unavailable')
class PartialTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def model(self,top=2):
        from transformers import WavLMConfig,WavLMModel
        from speakingt.partial_wavlm import Config,PartialWavLM
        c=WavLMConfig(hidden_size=16,intermediate_size=24,num_hidden_layers=12,num_attention_heads=2,
                      conv_dim=(8,8,8),conv_stride=(5,2,2),conv_kernel=(10,3,3),
                      num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,
                      hidden_dropout=0.,attention_dropout=0.,activation_dropout=0.,feat_proj_dropout=0.,
                      layerdrop=.9,mask_time_prob=0.)
        return PartialWavLM(Config(top_layers=top,hidden=8,dropout=0,warm_start=None),WavLMModel(c))

    def batch(self):
        import torch
        from speakingt.partial_wavlm import collate
        return collate([(torch.randn(640),torch.ones(4),'a'),(torch.randn(940),torch.ones(4)*2,'b')])

    def test_padding_and_native_single_agree(self):
        import torch
        m=self.model(); m.eval(); w,mask,_,_=self.batch()
        with torch.no_grad():
            pooled,_=m.features(w,mask)
            one=m.features(w[:1,:640],mask[:1,:640])[0]
            torch.testing.assert_close(pooled[:1],one,rtol=1e-4,atol=1e-5)
            bad=w.clone();bad[~mask]=999
            torch.testing.assert_close(m.features(bad,mask)[0],pooled,rtol=0,atol=0)

    def test_top_modes_gradients_and_frozen_state(self):
        import torch
        from speakingt.partial_wavlm import param_digest,make_optimizer,grad_report
        for top in (0,1,2):
            with self.subTest(top=top):
                m=self.model(top);m.train();w,mask,y,_=self.batch()
                self.assertFalse(m.wavlm.training)
                self.assertEqual([i for i,l in enumerate(m.wavlm.encoder.layers) if l.training],list(range(12-top,12)))
                frozen=param_digest(m)
                before={n:p.detach().clone() for n,p in m.wavlm.named_parameters() if p.requires_grad}
                opt=make_optimizer(m,m.cfg);((m(w,mask)-y)**2).mean().backward()
                info=grad_report(m)
                self.assertEqual(info['frozen_gradient_tensors'],0)
                self.assertEqual(info['encoder']['parameters']>0,top>0)
                if top:self.assertGreater(info['encoder']['norm'],0)
                opt.step();self.assertEqual(frozen,param_digest(m))
                if top:self.assertTrue(any(not torch.equal(before[n],p) for n,p in m.wavlm.named_parameters() if p.requires_grad))

    def test_masked_pooling_has_finite_zero_variance_gradient(self):
        import torch
        from speakingt.partial_wavlm import statistics
        x=torch.ones(2,4,3,requires_grad=True); mask=torch.tensor([[1,1,0,0],[1,1,1,0]],dtype=torch.bool)
        stats=statistics(x,mask);stats.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[~mask].abs().sum()),0)

    def test_invalid_mask_rejected(self):
        from speakingt.partial_wavlm import Config
        m=self.model();w,mask,_,_=self.batch();mask[0,2]=False
        with self.assertRaisesRegex(ValueError,'right-padded'):m(w,mask)
        for cfg in (Config(stage='full'),Config(top_layers=12),Config(precision='fp16'),Config(epochs=3),Config(head_lr=0)):
            with self.assertRaises(ValueError):cfg.validate()

    def test_row_membership_and_overlap(self):
        from speakingt import partial_wavlm as p
        rows=[{'id':'a','speaker':'s1','split':'train'}, {'id':'b','speaker':'s2','split':'validation'}]
        with patch.object(p,'load_manifest',return_value={'rows':rows}):
            p.validate_rows(rows,p.Config())
            with self.assertRaisesRegex(ValueError,'Duplicate'):p.validate_rows(rows+rows,p.Config())
            with self.assertRaisesRegex(ValueError,'non-development'):p.validate_rows([rows[0],dict(rows[1],id='test')],p.Config())
        leaked=[rows[0],dict(rows[1],speaker='s1')]
        with patch.object(p,'load_manifest',return_value={'rows':leaked}):
            with self.assertRaisesRegex(ValueError,'Speaker'):p.validate_rows(leaked,p.Config())

    def test_checkpoint_corruption_rejected_before_model_load(self):
        from speakingt import partial_wavlm as p
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt';p.atomic_checkpoint(path,{'schema':1})
            with path.open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'checksum'):p.restore(path)

    def test_accumulation_tail_matches_single_effective_batch(self):
        import torch
        from torch import nn
        from dataclasses import replace
        from speakingt import partial_wavlm as p
        class HeadOnly(nn.Module):
            def __init__(self):
                super().__init__();self.wavlm=nn.Linear(1,1);self.wavlm.requires_grad_(False);self.head=nn.Linear(1,4)
            def forward(self,w,mask):return self.head(w.mean(1,keepdim=True))
        items=[(torch.ones(500)*(i+1),torch.ones(4)*i,str(i)) for i in range(5)]
        a=HeadOnly();b=HeadOnly();b.load_state_dict(a.state_dict())
        cfg=p.Config(top_layers=0,batch_size=2,accumulation=3,clip_norm=1e6,warm_start=None)
        opta=p.make_optimizer(a,cfg);optb=p.make_optimizer(b,cfg)
        with patch.object(p,'WaveDataset',side_effect=lambda rows:items):
            p.train_epoch(a,items,cfg,opta,torch.amp.GradScaler('cuda',enabled=False),1)
            p.train_epoch(b,items,replace(cfg,batch_size=5,accumulation=1),optb,torch.amp.GradScaler('cuda',enabled=False),1)
        for x,y in zip(a.head.parameters(),b.head.parameters()):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)

    def test_fresh_statistics_use_training_rows_only(self):
        import torch
        from speakingt import partial_wavlm as p
        model=self.model()
        rows=[{'id':'a','split':'train','targets':[2.,3.,4.,5.]},
              {'id':'b','split':'validation','targets':[10.,10.,10.,10.]}]
        calls=[]
        def dataset(selected):
            calls.extend(r['id'] for r in selected)
            return [(torch.ones(640),torch.tensor(r['targets']),r['id']) for r in selected]
        values=torch.arange(12*32,dtype=torch.float32).reshape(1,12,32)
        with patch.object(p,'PartialWavLM',return_value=model),patch.object(p,'WaveDataset',side_effect=dataset), \
             patch.object(model,'features',return_value=(values,None)):
            result,provenance=p.initialize_model(model.cfg,rows)
        self.assertEqual(calls,['a'])
        torch.testing.assert_close(result.mean,values[0])
        torch.testing.assert_close(result.std,torch.full_like(result.std,1e-5))
        torch.testing.assert_close(result.head.network[-1].bias,torch.tensor(rows[0]['targets'])/10)
        self.assertEqual(provenance['initialization'],'fresh_head')

    def test_checkpoint_provenance_and_source_mismatch_rejected(self):
        from speakingt import partial_wavlm as p
        with tempfile.TemporaryDirectory() as d,patch.object(p,'data_identity',return_value={'revision':'expected'}):
            path=Path(d)/'last.pt'
            p.atomic_checkpoint(path,{'schema':p.SCHEMA,'identity':{'revision':'wrong'}})
            with self.assertRaisesRegex(ValueError,'provenance'):p.restore(path)
            p.atomic_checkpoint(path,{'schema':p.SCHEMA,'identity':{'revision':'expected'},'source_sha256':'wrong'})
            with self.assertRaisesRegex(ValueError,'Training code differs'):p.restore(path)

if __name__=='__main__':unittest.main()
