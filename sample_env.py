import gym

class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info

    def sample_pixel(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        # cur_obs = self.env.sim.render(64,64)
        cur_obs = self.env.render(mode='rgb_array')
        cur_state = self.current_state
        cur_obs = cur_obs.transpose((2,1,0))
        action = agent.select_action(cur_obs, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        # next_obs = self.env.sim.render(64,64)
        next_obs = self.env.render(mode='rgb_array')
        next_obs = next_obs.transpose((2,1,0))
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_obs, action, next_obs, reward, terminal, info
