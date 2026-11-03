"""Bounded raw-WAV partial fine-tuning CLI, shared with the Jupyter notebook."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt import partial_wavlm as p3

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','pilot','full','restore'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    parser.add_argument('--top-layers',type=int,choices=[0,1,2],default=2)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--encoder-lr',type=float,default=1e-5)
    parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--run-dir',help='New directory only; existing runs are never overwritten')
    parser.add_argument('--allow-full',action='store_true')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--warm-start',default=p3.WARM_RUN)
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    args=parser.parse_args()
    if args.stage=='restore':
        if not args.checkpoint: parser.error('--checkpoint required')
        model,saved,rows=p3.restore(args.checkpoint,args.device)
        result,_,_=p3.evaluate(model,[r for r in rows if r['split']=='validation'],saved['config']['batch_size'])
        print(result); return
    cfg=p3.Config(stage=args.stage,device=args.device,precision=args.precision,top_layers=args.top_layers,
                  epochs=args.epochs,batch_size=args.batch_size,accumulation=args.accumulation,
                  seed=args.seed,encoder_lr=args.encoder_lr,head_lr=args.head_lr,
                  patience=args.patience,threads=args.threads,
                  allow_full=args.allow_full,warm_start=None if args.fresh else args.warm_start)
    path=p3.run(cfg,run_dir=args.run_dir,resume=args.resume)
    print('Saved:',path)

if __name__=='__main__': main()
