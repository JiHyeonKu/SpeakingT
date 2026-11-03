"""Planning estimate from a completed real CUDA pilot; never rents or starts a GPU."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speakingt.partial_wavlm import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--hourly-usd',required=True,type=float);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--overhead-factor',type=float,default=2.)
    p.add_argument('--setup-hours',type=float,default=.25);args=p.parse_args()
    if args.hourly_usd<0 or args.epochs<1 or args.overhead_factor<1 or args.setup_hours<0:
        p.error('Invalid cost assumptions')
    cfg=read_json(args.run/'config.json')['config'];resources=read_json(args.run/'resources.json')
    summary=read_json(args.run/'summary.json');split=read_json(args.run/'split.json')
    if cfg['stage']!='pilot' or cfg['device']!='cuda' or summary['status']!='completed' or resources['cuda_peak_allocated_bytes']<=0:
        p.error('A completed, measured CUDA pilot is required; CPU timings are not GPU estimates')
    with (args.run/'metrics.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)<2 or int(rows[0]['epoch'])!=0:p.error('Need initial evaluation and a completed pilot epoch')
    seconds=(float(rows[-1]['elapsed_seconds'])-float(rows[0]['elapsed_seconds']))/(len(rows)-1)
    # Each epoch has a train pass, a separate train evaluation pass, and a validation pass.
    ratio=(2000*2+500)/(len(split['train'])*2+len(split['validation']))
    hours=args.setup_hours+args.overhead_factor*seconds*ratio*args.epochs/3600
    print({'measured_pilot_epoch_seconds':seconds,'pilot_peak_allocated_GiB':resources['cuda_peak_allocated_bytes']/2**30,
           'count_scaled_full_epoch_seconds':seconds*ratio,'planning_gpu_hours':hours,
           'planning_compute_usd':hours*args.hourly_usd,'epochs_assumed':args.epochs,
           'note':'Coarse count scaling only. Length, batch, precision, disk and GPU must be reviewed. Storage/transfer/tax extra. Approval required.'})


if __name__=='__main__':main()
