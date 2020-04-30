"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for Street Fighter""")
    parser.add_argument("--resume_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--num_actions", type=int, default=6)
    args = parser.parse_args()
    return args

def test(opt):
    torch.manual_seed(123)
    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path)
    env, num_states, num_actions = create_train_env(1, opt, "{}/test.mp4".format(opt.output_path))
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c".format(opt.resume_path)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c".format(opt.resume_path),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset(False, False, True))
    round_done, stage_done, game_done = False, False, True
    num_action = 0
    while True:
        if round_done or stage_done or game_done:
            h_0 = torch.zeros((1, 256), dtype=torch.float)
            c_0 = torch.zeros((1, 256), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        num_action += 1
        state, reward, round_done, stage_done, game_done = env.step(action)
        state = torch.from_numpy(state)
        if round_done or stage_done:
            state = torch.from_numpy(env.reset(round_done, stage_done, game_done))
        if game_done or num_action == opt.max_steps:
            env.make_anim()
            print("Game over")
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
