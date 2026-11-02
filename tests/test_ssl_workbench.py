"""Synthetic correctness tests; actual speech experiments are recorded separately."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

HAS_SSL=all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','psutil'))


class Phase2SplitTests(unittest.TestCase):
    def test_development_split_rejects_id_speaker_and_test_leakage(self):
        from speakingt.phase2_data import check_split
        check_split({'train_ids':['a'],'validation_ids':['b']},{'a':'s1','b':'s2'})
        with self.assertRaisesRegex(ValueError,'overlapping'):
            check_split({'train_ids':['a'],'validation_ids':['a']},{'a':'s1'})
        with self.assertRaisesRegex(ValueError,'Speaker'):
            check_split({'train_ids':['a'],'validation_ids':['b']},{'a':'s1','b':'s1'})
        with self.assertRaisesRegex(ValueError,'test contamination'):
            check_split({'train_ids':['a'],'validation_ids':['b']},{'a':'s1','b':'s2'},['b'])

    def test_waveform_preserves_samples_and_requires_original_format(self):
        from speakingt.phase2_data import read_audio
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'example.wav'
            pcm=np.arange(800,dtype=np.int16)
            with wave.open(str(path),'wb') as f:
                f.setparams((1,2,16000,0,'NONE','not compressed')); f.writeframes(pcm.tobytes())
            audio,digest=read_audio(path)
            np.testing.assert_array_equal(audio,pcm.astype(np.float32)/32768)
            self.assertEqual(len(digest),64)
            with wave.open(str(path),'wb') as f:
                f.setparams((1,2,8000,0,'NONE','not compressed')); f.writeframes(pcm.tobytes())
            with self.assertRaisesRegex(ValueError,'format'): read_audio(path)


@unittest.skipUnless(HAS_SSL,'optional SSL dependencies unavailable')
class SSLTests(unittest.TestCase):
    def test_extraction_fingerprint_ignores_comments_but_detects_code_changes(self):
        from unittest.mock import patch
        from speakingt import ssl_workbench as s
        with patch.object(s.inspect,'getsource',return_value='def extract(self, x):\n    # explanation\n    return x + 1\n'):
            first=s.extraction_code_hash()
        with patch.object(s.inspect,'getsource',return_value='def extract(self, x):\n    # revised explanation\n    return x + 1\n'):
            self.assertEqual(first,s.extraction_code_hash())
        with patch.object(s.inspect,'getsource',return_value='def extract(self, x):\n    return x + 2\n'):
            self.assertNotEqual(first,s.extraction_code_hash())

    def test_normalization_does_not_use_validation(self):
        from speakingt import ssl_workbench as s
        x=np.array([[[1.,3.]],[[3.,5.]],[[1000.,2000.]]],dtype=np.float32)
        normalized,mean,std=s.normalize_features(x,np.array([True,True,False]))
        np.testing.assert_allclose(mean,[[2.,4.]])
        np.testing.assert_allclose(std,[[1.,1.]])
        self.assertGreater(float(normalized[-1,0,0]),100)

    def test_metrics_units_and_constant_prediction_correlation(self):
        from speakingt import ssl_workbench as s
        y=np.array([[1,2,3,4],[3,4,5,6]],dtype=float)
        report=s.score_report(y+2,y)
        self.assertEqual(report['macro_mse'],4)
        self.assertEqual(report['macro_mae'],2)
        self.assertAlmostEqual(report['macro_pcc'],1)
        self.assertIsNone(s.score_report(np.ones((2,4)),y)['macro_pcc'])

    def test_feature_cache_rejects_provenance_or_tensor_change(self):
        from speakingt import ssl_workbench as s
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'feature.npz'
            x=np.ones((12,1536),dtype=np.float32)
            meta={'id':'a','feature_sha256':s._feature_digest(x)}
            np.savez(path,features=x,metadata=np.asarray(json.dumps(meta)))
            s.read_feature(path,{'id':'a'})
            with self.assertRaisesRegex(ValueError,'provenance'): s.read_feature(path,{'id':'b'})
            np.savez(path,features=x+1,metadata=np.asarray(json.dumps(meta)))
            with self.assertRaisesRegex(ValueError,'content'): s.read_feature(path,{'id':'a'})

    def test_real_backward_restore_and_no_overwrite(self):
        from speakingt import ssl_workbench as s
        rng=np.random.default_rng(6)
        n=12
        data={'features':rng.normal(size=(n,12,1536)).astype(np.float32),
              'targets':rng.uniform(2,9,(n,4)).astype(np.float32),
              'ids':np.array([str(i) for i in range(n)]),
              'speakers':np.array([f's{i//2}' for i in range(n)]),
              'split':np.array(['train']*8+['validation']*4)}
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            history=s.train(data,run,s.TrainConfig(epochs=2,hidden=8,dropout=0,batch_size=4,threads=1))
            self.assertGreater(history[-1]['weight_change_l2'],0)
            _,a=s.evaluate_checkpoint(run/'best.pt',data)
            _,b=s.evaluate_checkpoint(run/'best.pt',data)
            np.testing.assert_array_equal(a,b)
            self.assertFalse(json.loads((run/'summary.json').read_text())['test_evaluated'])
            with self.assertRaisesRegex(ValueError,'already exists'): s.train(data,run)
            with self.assertRaisesRegex(ValueError,'disabled'): s.evaluate_checkpoint(run/'best.pt',data,split='test')

    def test_invalid_configuration_and_speaker_overlap(self):
        from speakingt import ssl_workbench as s
        for cfg in (s.TrainConfig(epochs=0),s.TrainConfig(dropout=1),s.TrainConfig(learning_rate=float('nan'))):
            with self.assertRaises(ValueError): cfg.validate()
        data={'features':np.zeros((2,12,1536)), 'targets':np.ones((2,4)),
              'ids':np.array(['a','b']),'speakers':np.array(['s1','s1']),
              'split':np.array(['train','validation'])}
        with self.assertRaisesRegex(ValueError,'Speaker'): s.validate_dataset(data)

    def test_paired_speaker_bootstrap_keeps_models_paired(self):
        from speakingt.ssl_analysis import paired_bootstrap
        rng=np.random.default_rng(10)
        y=rng.uniform(1,9,(20,4)); p=y+rng.normal(0,.2,y.shape)
        speakers=np.repeat(['a','b','c','d'],5)
        intervals=paired_bootstrap(p,p,y,speakers,replicates=20)
        for target in intervals.values():
            for interval in target.values():
                self.assertEqual(interval['low'],0)
                self.assertEqual(interval['high'],0)

    def test_cache_only_rerun_preserves_manifest_fingerprint(self):
        from unittest.mock import patch
        from speakingt import ssl_workbench as s
        from speakingt.phase2_data import blob_sha,sha256
        class Encoder:
            provenance={'revision':'fixture'}
            calls=0
            def extract(self,samples):
                self.calls+=1
                return np.ones((12,1536),dtype=np.float32),{'input_samples':len(samples)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'audio').mkdir()
            path=root/'audio/a.wav'
            with wave.open(str(path),'wb') as f:
                f.setparams((1,2,16000,0,'NONE','not compressed'))
                f.writeframes(np.arange(800,dtype=np.int16).tobytes())
            row={'id':'a','audio_path':'audio/a.wav','git_blob_sha1':blob_sha(path.read_bytes())}
            manifest={'rows':[row]}
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            encoder=Encoder()
            with patch.object(s,'load_manifest',return_value=manifest):
                s.extract_features(encoder,directory=root)
                first=sha256(root/'feature-preview-1.json')
                s.extract_features(encoder,directory=root)
            self.assertEqual(encoder.calls,1)
            self.assertEqual(sha256(root/'feature-preview-1.json'),first)


if __name__=='__main__': unittest.main()
