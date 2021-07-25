import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools

device = torch.device('cuda')
BATCH_SIZE = 256

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):

    def __init__(self, hidden: int, output_size: int, input_width: int, input_height: int,
                 input_channel: int, ensemble_size: int, learning_rate=1e-3, weight_decay: float = 0., bias: bool = True):
        super(Encoder, self).__init__()

        self.hidden = hidden ## default hidden=32
        self.output_size = output_size ## state obs dim
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.ensemble_size = ensemble_size
        # self.weight_decay = weight_decay
        # self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=self.hidden, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden*2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.hidden*2, out_channels=self.hidden*4, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.hidden*4, out_channels=self.hidden*8, kernel_size=4, stride=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, obs):
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)

        self.linear = nn.Linear(x.shape[1], out_features=self.output_size)
        latent_state = self.linear(x)

        return latent_state

    def compile_transfer_loss(self, obs, action, reward, next_obs, dynamics_model, agent):

        latent = self.encode(obs)
        input = np.concatenate((latent, action), axis=-1)
        delta_latent, _ = dynamics_model.predict(input)
        predict_reward, delta_latent = delta_latent[:, 0], delta_latent[:, 1:]
        predict_latent = np.sum((latent, delta_latent), axis=-1)
        predict = np.concatenate((predict_reward, predict_latent), axis=-1)
        next_latent = self.encode(next_obs)
        label = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), next_latent), axis=-1)

        predict_action = agent.sample_action(predict_latent)

        latent_loss = nn.MSELoss(reduce=True, size_average=False)
        action_loss = nn.MSELoss(reduce=True, size_average=False)

        loss = latent_loss(predict, label) + action_loss(predict_action, action)

        return loss

    # def compile_encode_loss(self, input, label):

    def train_transfer(self, obs, action, reward, next_obs, dynamics_model, agent, decay_rate, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(obs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(obs.shape[0])
        obs, action, reward, next_obs = obs[permutation], action[permutation], reward[permutation], next_obs[permutation]

        train_obs, train_action, train_reward, train_next_obs = obs[num_holdout:], action[num_holdout:], reward[num_holdout:], next_obs[num_holdout:]
        holdout_obs, holdout_action, holdout_reward, holdout_next_obs = obs[:num_holdout], action[:num_holdout], reward[:num_holdout], next_obs[:num_holdout]

        holdout_obs = torch.from_numpy(holdout_obs).float().to(device)
        holdout_action = torch.from_numpy(holdout_action).float().to(device)
        holdout_reward = torch.from_numpy(holdout_reward).float().to(device)
        holdout_next_obs = torch.from_numpy(holdout_next_obs).float().to(device)

        for epoch in itertools.count():
            for start_pos in range(0, train_obs.shape[0], batch_size):
                train_obs_ = torch.from_numpy(train_obs[start_pos: start_pos + batch_size]).float().to(device)
                train_action_ = torch.from_numpy(train_action[start_pos: start_pos + batch_size]).float().to(device)
                train_reward_ = torch.from_numpy(train_reward[start_pos: start_pos + batch_size]).float().to(device)
                train_next_obs_ = torch.from_numpy(train_next_obs[start_pos: start_pos + batch_size]).float().to(device)

                losses = []
                loss = self.compile_transfer_loss(train_obs_, train_action_, train_reward_, train_next_obs_, dynamics_model, agent)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels,
                                                                 inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False





    # def train(self, obs, action, reward, next_obs, policy, decay_rate):

