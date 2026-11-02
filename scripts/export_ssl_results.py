"""Export aggregate public results, never raw recordings or per-utterance scores."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from speakingt.phase2_data import DATA, sha256, write_json, preserve_baseline


def export(comparison, output):
    comparison,output=Path(comparison),Path(output)
    summary=json.loads((comparison/'summary.json').read_text(encoding='utf-8'))
    if summary['test_evaluated'] or sorted(r['seed'] for r in summary['records'])!=[7,19,42]:
        raise ValueError('Expected completed three-seed development experiment')
    if any(r['summary']['status'] not in ('completed','early_stopped') for r in summary['records']):
        raise ValueError('An experiment did not finish')
    preserve_baseline()
    output.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','comparison.csv','comparison.png'):
        shutil.copyfile(comparison/name,output/name)
    audit=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['audit']
    audio=json.loads((DATA/'audio-audit.json').read_text(encoding='utf-8'))
    features=json.loads((DATA/'feature-manifest.json').read_text(encoding='utf-8'))
    write_json(output/'data-readiness.json',{
        'audit':audit,'wavs_verified':audio['count'],'duration_seconds':audio['duration_seconds'],
        'exact_pcm_duplicate_groups':audio['decoded_duplicate_groups'],
        'feature_count':features['count'],'feature_extraction_resources':features['resources'],
        'feature_extraction_newly_computed':features['computed'],
        'encoder':features['encoder'],
        'manifest_sha256':sha256(DATA/'manifest.json'),
        'feature_manifest_sha256':sha256(DATA/'feature-manifest.json'),
        'note':'Whole extraction job reused 20 already computed Stage B items; resources do not include earlier jobs.'})
    for name in ('stage-a.json','stage-b.json'):
        shutil.copyfile(ROOT/'reports/phase2'/name,output/name)
    artifacts={}
    for record in summary['records']:
        run=ROOT/record['run']; seed=record['seed']
        for name in ('learning-curves.png','metrics.csv'):
            shutil.copyfile(run/name,output/f'seed-{seed}-{name}')
        artifacts[str(seed)]={
            'run':record['run'],
            'files':{name:sha256(run/name) for name in
                     ('best.pt','initial.pt','last.pt','config.json','split.json','summary.json','metrics.csv')},
            'source_sha256':json.loads((run/'config.json').read_text(encoding='utf-8'))['source_sha256']}
    notebook=ROOT/'reports/phase2/notebook-full-verified.ipynb'
    executed=json.loads(notebook.read_text(encoding='utf-8'))
    cells=[c for c in executed['cells'] if c['cell_type']=='code']
    if any(c['execution_count'] is None or any(o['output_type']=='error' for o in c['outputs']) for c in cells):
        raise ValueError('Full notebook was not successfully executed')
    write_json(output/'artifacts.json',{
        'runs':artifacts,'executed_notebook':notebook.relative_to(ROOT).as_posix(),
        'executed_notebook_sha256':sha256(notebook),'executed_code_cells':len(cells),
        'notebook_source_sha256':sha256(ROOT/'notebooks/Phase2_Frozen_WavLM.ipynb'),
        'protocol_sha256':sha256(ROOT/'experiments/phase2/protocol.json'),
        'official_test_evaluated':False})
    print('Exported aggregates and graphs:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison')
    parser.add_argument('--output',default='experiments/phase2/results')
    args=parser.parse_args()
    export(ROOT/args.comparison,ROOT/args.output)
