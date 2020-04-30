"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.model import ActorCritic, IntrinsicCuriosityModule
from src.optimizer import GlobalAdam
from src.process import local_train
import torch.multiprocessing as _mp
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for Street Fighter""")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--sigma', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--lambda_', type=float, default=0.1, help='a3c loss coefficient')
    parser.add_argument('--eta', type=float, default=0.2, help='intrinsic coefficient')
    parser.add_argument('--beta', type=float, default=0.2, help='curiosity coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50, help="Number of steps between updates")
    parser.add_argument("--num_global_steps", type=int, default=1e8)
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--save_interval", type=int, default=10000, help="Number of steps between savings")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--exp", type=str, default="prior_knowledge", help="Desired name for the experiment")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=6)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    args = parser.parse_args()
    return args


def train(opt):
    #torch.manual_seed(123)
    opt.log_path = opt.log_path + "/" + opt.exp
    opt.saved_path = opt.saved_path + "/" + opt.exp
    opt.output_path = opt.output_path + "/" + opt.exp
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    mp = _mp.get_context("spawn")
    global_model = ActorCritic(num_inputs=3, num_actions=opt.num_actions)
    global_icm = IntrinsicCuriosityModule(num_inputs=3, num_actions=opt.num_actions)

    if opt.resume_path:
        print("Load model from checkpoint: {}".format(opt.resume_path))
        global_model.load_state_dict(torch.load("{}/a3c".format(opt.resume_path)))
        global_icm.load_state_dict(torch.load("{}/icm".format(opt.resume_path)))

    if opt.use_gpu:
        global_model.cuda()
        global_icm.cuda()
    global_model.share_memory()
    global_icm.share_memory()

    optimizer = GlobalAdam(list(global_model.parameters()) + list(global_icm.parameters()), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
