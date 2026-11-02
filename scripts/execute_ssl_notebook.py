"""Execute the actual notebook in a Jupyter kernel; save outputs outside Git."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key,relative in [('IPYTHONDIR','.cache/ipython'),('JUPYTER_CONFIG_DIR','.jupyter/config'),
                     ('JUPYTER_RUNTIME_DIR','.jupyter/runtime'),('MPLCONFIGDIR','.cache/matplotlib')]:
    os.environ[key]=str(ROOT/relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Actually train 2000/500 with all three predeclared seeds')
    parser.add_argument('--wait-for-features',action='store_true',help='Wait for the independent frozen-feature job to finish first')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists(): raise ValueError('Executed notebook already exists; use a new output path')
    if args.wait_for_features:
        import time
        print('Waiting for complete 2500-item feature manifest before notebook execution.',flush=True)
        while not (ROOT/'data/phase2/feature-manifest.json').exists():
            time.sleep(5)
    if args.full: os.environ['SPEAKINGT_SSL_FULL']='1'
    else: os.environ.pop('SPEAKINGT_SSL_FULL',None)
    import nbformat
    from nbclient import NotebookClient
    notebook=nbformat.read(ROOT/'notebooks/Phase2_Frozen_WavLM.ipynb',as_version=4)
    def start(cell,cell_index,**kwargs):
        if cell.cell_type=='code': print(f'Executing notebook cell {cell_index}',flush=True)
    client=NotebookClient(notebook,timeout=3600,kernel_name='python3',
                          resources={'metadata':{'path':str(ROOT)}},on_cell_start=start)
    try:
        client.execute()
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(notebook,output)
    print('Completed actual notebook:',output,flush=True)


if __name__=='__main__': main()
