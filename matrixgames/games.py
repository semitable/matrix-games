import numpy as np
import gymnasium as gym


def actions_to_onehot(num_actions, actions):
    """
    Transfer actions to onehot representation
    :param num_actions: list of number of actions of each agent
    :param actions: list of actions (int) for each agent
    :return: onehot representation of actions
    """
    onehot = [[0] * num_action for num_action in num_actions]
    for ag, act in enumerate(actions):
        onehot[ag][act] = 1
    return onehot


class MatrixGame(gym.Env):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 5,
    }

    def __init__(self, payoff_matrix, ep_length, last_action_state=True):
        """
        Create matrix game
        :param payoff_matrix: list of lists or numpy array for payoff matrix of all agents
        :param ep_length: length of episode (before done is True)
        :param last_action_state: boolean flag indicating whether last actions should be returned
                                  as state of the environment or just 0s for all agents
        """
        self.payoff = payoff_matrix
        self.n_agents = self.payoff.shape[0]
        self.num_actions = [self.payoff.shape[i + 1] for i in range(self.n_agents)]

        self.ep_length = ep_length
        self.last_action_state = last_action_state

        self.last_actions = [-1 for _ in range(self.n_agents)]
        self.t = 0

        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(num_action) for num_action in self.num_actions]
        )

        shape = (self.n_agents,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

    def _make_obs(self):
        if self.last_action_state:
            return tuple([np.array(self.last_actions)] * self.n_agents)
        else:
            return tuple([np.array(self.n_agents * [0])] * self.n_agents)

    def reset(self, seed=None, options=None):
        self.t = 0
        # self.last_actions = actions_to_onehot(self.num_actions, [0] * self.n_agents)
        return self._make_obs(), {}

    def step(self, action):
        self.t += 1
        self.last_actions = actions_to_onehot(self.num_actions, action)
        reward = self.payoff

        rewards = [0 for _ in range(len(action))]
        for i in range(len(action)):
            temp = self.payoff[i]
            for j in range(len(action)):
                temp = temp[action[j]]
            rewards[i] = temp

        done = self.t >= self.ep_length
        truncated = False

        return self._make_obs(), rewards, done, truncated, {}

    def render(self):
        if self.last_action_state:
            out = " | ".join([f"Agent {i + 1}: {self.last_actions[i]}" for i in range(self.n_agents)])
        else:
            out = f"Step {self.t}"
        return out


class TwoStep(gym.Env):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 5,
    }

    def __init__(self, payoff_matrix1, payoff_matrix2):
        self.payoff1 = payoff_matrix1
        self.payoff2 = payoff_matrix1

        self.matrix1 = MatrixGame(payoff_matrix1, ep_length=1, last_action_state=False)
        self.matrix2 = MatrixGame(payoff_matrix2, ep_length=1, last_action_state=False)
        self.n_agents = self.matrix1.n_agents

        self.ep_length = 2

        self.action_space = self.matrix1.action_space
        shape = (3,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

        self.t = 0
        self.state = "A"

    def _make_obs(self):
        if self.state == "A":
            x = [1, 0, 0]
        elif self.state == "2A":
            x = [0, 1, 0]
        else:
            x = [0, 0, 1]
        return tuple([np.array(x)] * self.n_agents)

    def reset(self, seed=None, options=None):
        self.state = "A"
        self.t = 0
        self.matrix1.reset()
        self.matrix2.reset()

        return self._make_obs(), {}

    def step(self, action):
        if self.t == 0:
            if action[0] == 0:
                self.state = "2A"
            else:
                self.state = "2B"
            rewards = self.n_agents * [0]
        elif self.t == 1:
            if self.state == "2A":
                _, rewards, _, _, _ = self.matrix1.step(action)
            else:
                # state "2B"
                _, rewards, _, _, _ = self.matrix2.step(action)
        else:
            rewards = self.n_agents * [0]

        done = self.t != 0
        truncated = False
        self.t += 1

        return self._make_obs(), rewards, done, truncated, {}

    def render(self):
        return f"Step {self.t} at state {self.state}"


# penalty game
def create_penalty_game(penalty, ep_length, last_action_state=True):
    assert penalty <= 0
    payoff = [
        [penalty, 0, 10],
        [0, 2, 0],
        [10, 0, penalty],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game


# climbing game
def create_climbing_game(ep_length, last_action_state=True):
    payoff = [
        [11, -30, 0],
        [-30, 7, 0],
        [0, 6, 5],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game
