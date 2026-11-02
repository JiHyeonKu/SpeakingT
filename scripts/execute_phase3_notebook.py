"""Execute the actual Phase 3 notebook or a fresh-kernel restore verification."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key,relative in [('IPYTHONDIR','.cache/ipython'),('JUPYTER_CONFIG_DIR','.jupyter/config'),
                     ('JUPYTER_RUNTIME_DIR','.jupyter/runtime'),('MPLCONFIGDIR','.cache/matplotlib')]:
    os.environ[key]=str(ROOT/relative)
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--restore-run')
args=parser.parse_args();output=ROOT/args.output
if output.exists():raise ValueError('Output already exists')
nb=nbformat.read(ROOT/'notebooks/Phase3_Partial_WavLM.ipynb',as_version=4)
if args.restore_run:
    env=next(c for c in nb.cells if c.cell_type=='code' and 'from speakingt import partial_wavlm' in c.source)
    restore=next(c for c in nb.cells if c.cell_type=='code' and "RESTORE_RUN = ''" in c.source)
    restore.source=restore.source.replace("RESTORE_RUN = ''",f'RESTORE_RUN = {args.restore_run!r}')
    nb.cells=[nbformat.v4.new_markdown_cell('Fresh-kernel restore check'),env,
              nbformat.v4.new_markdown_cell('Regenerate saved predictions without training'),restore,
              nbformat.v4.new_code_cell("with np.load(restore_run/'last-validation-predictions.npz') as prior:\n    np.testing.assert_array_equal(predictions,prior['predictions'])\nprint('Fresh kernel predictions bitwise identical')")]
def start(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print('Executing cell',cell_index,flush=True)
try:
    NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=start).execute()
finally:
    output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,output)
print('Completed:',output,flush=True)
