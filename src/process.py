"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import torch
from src.env import create_train_env
from src.model import ActorCritic, IntrinsicCuriosityModule
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import timeit


def local_train(index, opt, global_model, global_icm, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(index+1, opt, "{}/test.mp4".format(opt.output_path))
    local_model = ActorCritic(num_states, num_actions)
    local_icm = IntrinsicCuriosityModule(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
        local_icm.cuda()
    local_model.train()
    local_icm.train()
    inv_criterion = nn.CrossEntropyLoss()
    fwd_criterion = nn.MSELoss()
    state = torch.from_numpy(env.reset(False, False, True))
    if opt.use_gpu:
        state = state.cuda()
    round_done, stage_done, game_done = False, False, True
    curr_step = 0
    total_step = 0
    curr_episode = 0
    return_eps = 0
    next_save = False
    while True:
        if save and next_save:
            next_save = False
            saved_path = opt.saved_path + "/" + str(total_step // 1000) + "K"
            if not os.path.isdir(saved_path):
                os.makedirs(saved_path)
            torch.save(global_model.state_dict(),
                       "{}/a3c".format(saved_path))
            torch.save(global_icm.state_dict(),
                           "{}/icm".format(saved_path))
        #curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        local_icm.load_state_dict(global_icm.state_dict())
        if round_done or stage_done or game_done:
            h_0 = torch.zeros((1, 256), dtype=torch.float)
            c_0 = torch.zeros((1, 256), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []
        inv_losses = []
        fwd_losses = []
        action_cnt = [0] * num_actions
        highest_position = env.game.newGame.Players[0].getPosition()
        first_policy = None

        for i in range(opt.num_local_steps):
            total_step += 1
            if total_step % opt.save_interval == 0 and total_step > 0: next_save = True
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()
            action_cnt[action] += 1
            if i == 0:
                first_policy = policy

            next_state, reward, round_done, stage_done, game_done = env.step(action)
            return_eps += reward
            highest_position = max(highest_position, env.game.newGame.Players[0].getPosition(), key=lambda p: -p[1])
            next_state = torch.from_numpy(next_state)
            if opt.use_gpu:
                next_state = next_state.cuda()
            action_oh = torch.zeros((1, num_actions))  # one-hot action
            action_oh[0, action] = 1
            if opt.use_gpu:
                action_oh = action_oh.cuda()
            pred_logits, pred_phi, phi = local_icm(state, next_state, action_oh)
            if opt.use_gpu:
                inv_loss = inv_criterion(pred_logits, torch.tensor([action]).cuda())
            else:
                inv_loss = inv_criterion(pred_logits, torch.tensor([action]))
            fwd_loss = fwd_criterion(pred_phi, phi) / 2
            intrinsic_reward = opt.eta * fwd_loss.detach()
            reward += intrinsic_reward

            if curr_step >= opt.max_steps:
                round_done, stage_done, game_done = False, False, True

            if round_done or stage_done or game_done:
                curr_step = 0
                curr_episode += 1
                next_state = torch.from_numpy(env.reset(round_done, stage_done, game_done))
                if opt.use_gpu:
                    next_state = next_state.cuda()
                if save: 
                    writer.add_scalar("Train_{}/Return".format(index), return_eps, curr_episode)
                return_eps = 0

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)
            inv_losses.append(inv_loss)
            fwd_losses.append(fwd_loss)
            state = next_state
            if round_done or stage_done or game_done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not (round_done or stage_done or game_done):
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        curiosity_loss = 0
        next_value = R

        for value, log_policy, reward, entropy, inv, fwd in list(
                zip(values, log_policies, rewards, entropies, inv_losses, fwd_losses))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            curiosity_loss = curiosity_loss + (1 - opt.beta) * inv + opt.beta * fwd

        total_loss = opt.lambda_ * (-actor_loss + critic_loss - opt.sigma * entropy_loss) + curiosity_loss
        if save:
            writer.add_scalar("Train_{}/Loss".format(index), total_loss, total_step)
            c_loss = curiosity_loss.item()
            t_loss = total_loss.item()
            print("Process {}. Episode {}. A3C Loss: {}. ICM Loss: {}.".
		  format(index, curr_episode, t_loss - c_loss, c_loss))
            print("# Actions Tried: {}. Highest Position: {}. First Policy: {}".
                  format(action_cnt, highest_position, first_policy.cpu().detach().numpy()))
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        for local_param, global_param in zip(local_icm.parameters(), global_icm.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return

