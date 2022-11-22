import os
import argparse
from evaluation import dtu_eval, epfl_eval
from pathlib import Path
from pyhocon import ConfigFactory


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default='./confs/base.conf')
parser.add_argument('--case', type=str, default='')
parser.add_argument("--suffix", default="")

args = parser.parse_args()
dataset = args.case.split('/')[0]
scene = args.case.split('/')[-1].replace('scan', '')

exp_dir = Path("exp/{}".format(args.case))

eval_dir = Path("evals/{}".format(args.case))
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

inp_mesh_path = Path(f'{exp_dir}/meshes/output_mesh{args.suffix}.ply')


if "dtu" in dataset or dataset == "DTU":
    dtu_eval.eval(inp_mesh_path, int(scene), "/mnt/D/hust_fu/Data/dtu_eval", eval_dir, args.suffix)
else:
    epfl_eval.eval(inp_mesh_path, scene, "data/epfl", eval_dir, args.suffix)