"""Publish only aggregate CPU verification artifacts, never individual predictions/audio/weights."""
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt.phase2_data import ROOT,sha256,write_json

RUNS={'initial':'20261102-172648-542194','resumed':'20261103-090743-101198',
      'continuous':'20261103-090843-541144'}


def main():
    out=ROOT/'experiments/phase3';out.mkdir(parents=True,exist_ok=True)
    summary={}
    for label,name in RUNS.items():
        source=ROOT/'runs/phase3'/name;destination=out/label;destination.mkdir(exist_ok=True)
        files=['config.json','summary.json','resources.json','metrics.csv','best-metrics.json','learning-curves.png']
        files.extend(p.name for p in sorted(source.glob('metrics-*.json')))
        for filename in files:shutil.copyfile(source/filename,destination/filename)
        cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
        summary[label]={'run':str(source.relative_to(ROOT)).replace('\\','/'),
                        'best_checkpoint_sha256':sha256(source/'best.pt'),'last_checkpoint_sha256':sha256(source/'last.pt'),
                        'source_sha256':cfg['source_sha256']}
    checks=[]
    for filename in ('notebook-stage-ab.ipynb','notebook-fresh-restore.ipynb'):
        path=ROOT/'reports/phase3'/filename;notebook=json.loads(path.read_text(encoding='utf-8'))
        outputs=[o for c in notebook['cells'] for o in c.get('outputs',[])]
        assert not any(o.get('output_type')=='error' for o in outputs)
        executed=sum(c['cell_type']=='code' and c.get('execution_count') is not None for c in notebook['cells'])
        if 'fresh' in filename:
            assert any('Fresh kernel predictions bitwise identical' in ''.join(o.get('text',[])) for o in outputs)
        checks.append({'file':filename,'sha256':sha256(path),'executed_code_cells':executed,'errors':0})
    tests=ROOT/'reports/phase3/tests-20261103.txt';text=tests.read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',text)
    assert match,'Full test suite did not pass'
    write_json(out/'verification.json',{'verified_date':'2026-11-03','runs':summary,'notebooks':checks,
        'tests':{'passed':int(match[1]),'seconds':float(match[2]),'log_sha256':sha256(tests)},
        'gpu_executed':False,'paid_compute_usd':0,'original_test_used':False})
    print('Exported aggregate results only:',out)


if __name__=='__main__':main()
