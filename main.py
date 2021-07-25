import argparse
import time
import gym
import torch
import numpy as np
from itertools import count

import logging

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC, TransferPixelSAC, PixelSAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
import matplotlib.pyplot as plt

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=1, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=1, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--pixel_policy_train_batch_size', type=int, default=64, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--init_pixel_exploration_steps', type=int, default=1000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--model_type', default='tensorflow', metavar='A',
                        help='predict model -- pytorch or tensorflow')
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--model_dir', default='./model_file/',
                        help='your model save path')
    parser.add_argument('--model_name', default='model.pt',
                        help='your model save path')
    parser.add_argument('--input_type', default='state',
                        help='input type can be state or pixels')
    parser.add_argument('--is_transfer', default=False,
                        help='only effective when the input type is pixel')
    parser.add_argument('--obs_shape', default=64)
    return parser.parse_args()

from PIL import Image

def train(args, env_sampler, predict_env, agent, env_pool, model_pool, model_dir, input_type, is_transfer, dynamics_model):
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    print('start exploration')
    exploration_before_start(args, env_sampler, env_pool, agent, input_type)
    print(len(env_pool))
    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step
            # print('cur_step:', cur_step, 'start_step:', start_step, 'epoch_length:', args.epoch_length)
            if cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size:
                break
            if input_type == 'state':
                if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                    # if cur_step == 1:
                        # img = env.sim.render(64, 64)
                        # image = Image.fromarray(img)
                        # image.save('./e.jpg')
                        # print('image saved')
                    # print('state')
                    ## 训练模型
                    train_predict_model(args, env_pool, predict_env)

                    if cur_step % 100000 == 0 and cur_step > 0:
                        torch.save({'Dynamics':predict_env.model.ensemble_model.state_dict()}, model_dir+'step-{}-model.pt'.format(cur_step))

                    new_rollout_length = set_rollout_length(args, epoch_step)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        model_pool = resize_model_pool(args, rollout_length, model_pool)
                    ## rollout生成样本池训练policy
                    rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

                cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
                env_pool.push(cur_state, action, reward, next_state, done)

                if len(env_pool) > args.min_pool_size:
                    train_policy_steps += train_state_policy_repeats(args, total_step, train_policy_steps, cur_step,
                                                               env_pool, model_pool, agent)

                total_step += 1

                if total_step % 1000 == 0:
                    '''
                    avg_reward_len = min(len(env_sampler.path_rewards), 5)
                    avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                    logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                    print(total_step, env_sampler.path_rewards[-1], avg_reward)
                    '''
                    env_sampler.current_state = None
                    sum_reward = 0
                    done = False
                    while not done:
                        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                        sum_reward += reward
                    # logger.record_tabular("total_step", total_step)
                    # logger.record_tabular("sum_reward", sum_reward)
                    # logger.dump_tabular()
                    logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                    print(total_step, sum_reward)
            elif input_type == 'pixel':
                if cur_step > 0 and cur_step % args.model_train_freq == 0:
                    if is_transfer:
                        if len(env_pool) > args.min_pool_size:
                            train_policy_steps += train_transfer_pixel_policy_repeats(args, total_step, train_policy_steps, cur_step,
                                                                       env_pool, agent, dynamics_model)
                    else:
                        if len(env_pool) > args.min_pool_size:
                            train_policy_steps += train_pixel_policy_repeats(args, total_step, train_policy_steps, cur_step,
                                                                       env_pool, agent)

                cur_obs, action, next_obs, reward, done, info = env_sampler.sample_pixel(agent)
                env_pool.push_pixel(cur_obs, action, reward, next_obs, done)

                total_step += 1

                if total_step % 1000 == 0:
                    '''
                    avg_reward_len = min(len(env_sampler.path_rewards), 5)
                    avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                    logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                    print(total_step, env_sampler.path_rewards[-1], avg_reward)
                    '''
                    env_sampler.current_state = None
                    sum_reward = 0
                    done = False
                    while not done:
                        cur_state, action, next_state, reward, done, info = env_sampler.sample_pixel(agent,
                                                                                               eval_t=True)
                        sum_reward += reward
                    # logger.record_tabular("total_step", total_step)
                    # logger.record_tabular("sum_reward", sum_reward)
                    # logger.dump_tabular()
                    logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                    print(total_step, sum_reward)
            else:
                raise ValueError('input type can only be "state" or "pixels", got %s.' % input_type)
            # if input_type == 'state':
            #     cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            #     env_pool.push(cur_state, action, reward, next_state, done)
            # elif input_type == 'pixel':
            #     cur_obs, action, next_obs, reward, done, info = env_sampler.sample_pixel(agent)
            #     env_pool.push_pixel(cur_obs, action, reward, next_obs, done)
            # else:
            #     raise ValueError('input type can only be "state" or "pixels", got %s.' % input_type)
            # if cur_step==1:
            #     # env.sim.render(mode='window', width=16, height=16)
            #     img = env.render(mode="rgb_array") ### (500, 500, 3)
            #     print(img.shape)
            #     # Img = Image.fromarray(img)
            #     # Img.save('./e.png')

            #
            # if len(env_pool) > args.min_pool_size:
            #     train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
            #
            # total_step += 1
            #
            # if total_step % 1000 == 0:
            #     '''
            #     avg_reward_len = min(len(env_sampler.path_rewards), 5)
            #     avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
            #     logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
            #     print(total_step, env_sampler.path_rewards[-1], avg_reward)
            #     '''
            #     env_sampler.current_state = None
            #     sum_reward = 0
            #     done = False
            #     while not done:
            #         cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
            #         sum_reward += reward
            #     # logger.record_tabular("total_step", total_step)
            #     # logger.record_tabular("sum_reward", sum_reward)
            #     # logger.dump_tabular()
            #     logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
            #     print(total_step, sum_reward)


def exploration_before_start(args, env_sampler, env_pool, agent, input_type):
    if input_type == 'state':
        for i in range(args.init_exploration_steps):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)
    elif input_type == 'pixel':
        for i in range(args.init_pixel_exploration_steps):
            print(i)
            cur_obs, action, next_obs, reward, done, info = env_sampler.sample_pixel(agent)
            env_pool.push_pixel(cur_obs, action, reward, next_obs, done)
    else:
        raise ValueError('input type can only be "state" or "pixels", got %s.' % input_type)

def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_state_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


def train_pixel_policy_repeats(args, total_step, train_step, cur_step, env_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.pixel_policy_train_batch_size)

        env_obs, env_action, env_reward, env_next_obs, env_done = env_pool.sample(int(env_batch_size))

        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = env_obs, env_action, env_reward, env_next_obs, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_obs, batch_action, batch_reward, batch_next_obs, batch_done), args.pixel_policy_train_batch_size, i)

    return args.num_train_repeat


def train_transfer_pixel_policy_repeats(args, total_step, train_step, cur_step, env_pool, agent, dynamics_model):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.pixel_policy_train_batch_size)

        env_obs, env_action, env_reward, env_next_obs, env_done = env_pool.sample(int(env_batch_size))

        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = env_obs, env_action, env_reward, env_next_obs, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_obs, batch_action, batch_reward, batch_next_obs, batch_done), args.pixel_policy_train_batch_size, i)

    return args.num_train_repeat


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    obs_shape = env.render(mode='rgb_array').shape ### (width, height, channel)

    # Intial agent
    if args.input_type == 'state':
        state_size = np.prod(env.observation_space.shape)
        action_size = np.prod(env.action_space.shape)
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size,
                                              args.reward_size, args.pred_hidden_size,
                                              use_decay=args.use_decay)
        agent = SAC(num_inputs=env.observation_space.shape[0], action_space=env.action_space, args=args)
        # Predict environments
        predict_env = PredictEnv(env_model, args.env_name, args.model_type)

        # Initial pool for env
        env_pool = ReplayMemory(args.replay_size)
        # Initial pool for model
        rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
        model_steps_per_epoch = int(1 * rollouts_per_epoch)
        new_pool_size = args.model_retain_epochs * model_steps_per_epoch
        model_pool = ReplayMemory(new_pool_size)
        model_dir = args.model_dir
        input_type = args.input_type
        is_transfer = args.is_transfer
        # Sampler of environment
        env_sampler = EnvSampler(env)

        train(args, env_sampler, predict_env, agent, env_pool, model_pool, model_dir, input_type, is_transfer)
    elif args.input_type == 'pixel':
        if args.is_transfer:
            state_size = np.prod(env.observation_space.shape)
            action_size = np.prod(env.action_space.shape)
            transfer_env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
            state_dict = torch.load(args.model_dir+args.model_path)
            transfer_env_model.load_state_dict(state_dict['Dynamics'])
            agent = TransferPixelSAC(input_channel=obs_shape[-1],linear_inputs_dim=env.observation_space.shape[0],
                                     linear_hidden_dim=args.hidden_size,action_dim=env.action_space.shape[0],
                                     action_space=env.action_space,dynamics_model=transfer_env_model,args=args)
            # Predict environments
            predict_env = None

            # Initial pool for env
            env_pool = ReplayMemory(args.replay_size)
            # Initial pool for model
            rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
            model_steps_per_epoch = int(1 * rollouts_per_epoch)
            new_pool_size = args.model_retain_epochs * model_steps_per_epoch
            model_pool = ReplayMemory(new_pool_size)
            model_dir = args.model_dir
            input_type = args.input_type
            is_transfer = args.is_transfer
            # Sampler of environment
            env_sampler = EnvSampler(env)

            train(args, env_sampler, predict_env, agent, env_pool, model_pool, model_dir, input_type, is_transfer, transfer_env_model)
        else:
            agent = PixelSAC(input_channel=obs_shape[-1], linear_inputs_dim=env.observation_space.shape[0],
                                     linear_hidden_dim=args.hidden_size, action_dim=env.action_space.shape[0],
                                     action_space=env.action_space, args=args)
            # Predict environments
            predict_env = None

            # Initial pool for env
            env_pool = ReplayMemory(args.replay_size)
            # Initial pool for model
            rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
            model_steps_per_epoch = int(1 * rollouts_per_epoch)
            new_pool_size = args.model_retain_epochs * model_steps_per_epoch
            model_pool = ReplayMemory(new_pool_size)
            model_dir = args.model_dir
            input_type = args.input_type
            is_transfer = args.is_transfer
            # Sampler of environment
            env_sampler = EnvSampler(env)

            train(args, env_sampler, predict_env, agent, env_pool, model_pool, model_dir, input_type, is_transfer)
    else:
        raise ValueError('input type can only be "state" or "pixels", got %s.' % args.input_type)


if __name__ == '__main__':
    main()
