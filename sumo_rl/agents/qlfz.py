import numpy as np
import random
import gymnasium as gym
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from .ql_agent import QLAgent  # 假设原来的 QLAgent 在 your_module 中定义


class FuzzyQLAgent(QLAgent):
    """
    在原有 QLAgent 基础上加入模糊逻辑：
     - 对 state 空间中指定的维度做模糊化
     - 使用模糊状态 tuple 作为 Q-table 的 key
    """

    def __init__(
        self,
        starting_state,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        fuzzy_dims: list[int],
        alpha=0.5,
        gamma=0.95,
        exploration_strategy=EpsilonGreedy(),
    ):
        super().__init__(starting_state, state_space, action_space, alpha, gamma, exploration_strategy)

        self.fuzzy_dims = fuzzy_dims

        self.fuzzy_vars: dict[int, ctrl.Antecedent] = {}
        self.labels = ['low', 'medium', 'high']
        for d in self.fuzzy_dims:
            low, high = 0, 1 # state space is normalized
            universe = np.linspace(low, high, 100)
            var = ctrl.Antecedent(universe, f's{d}')

            mid = (low + high) / 2
            var['low']    = fuzz.trimf(universe, [low, low, mid])
            var['medium'] = fuzz.trimf(universe, [low, mid, high])
            var['high']   = fuzz.trimf(universe, [mid, high, high])

            self.fuzzy_vars[d] = var

    def _fuzzy_state(self, raw_state: np.ndarray) -> tuple[str, ...]:
        """
        将连续状态 raw_state 映射到一个模糊状态 tuple，
         ('low', 'high', 'medium')。
        """
        labels = []
        for d in self.fuzzy_dims:
            val = raw_state[d]
            var = self.fuzzy_vars[d]
            m = {lbl: fuzz.interp_membership(var.universe, var[lbl].mf, val) 
                 for lbl in self.labels}
            labels.append(max(m, key=m.get))
        return tuple(labels)

    def act(self) -> int:
        """
        覆盖父类 act：根据模糊状态做 ε-greedy 选择
        """
        fuzzy_s = self._fuzzy_state(self.state)
        if fuzzy_s not in self.q_table:
            self.q_table[fuzzy_s] = [0.0] * self.action_space.n

        self.action = self.exploration.choose(self.q_table, fuzzy_s, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        """
        覆盖父类 learn：在模糊状态空间上更新 Q 值
        """
        fuzzy_s  = self._fuzzy_state(self.state)
        fuzzy_s1 = self._fuzzy_state(next_state)

        if fuzzy_s1 not in self.q_table:
            self.q_table[fuzzy_s1] = [0.0] * self.action_space.n

        a = self.action
        q_sa = self.q_table[fuzzy_s][a]
        q_next_max = max(self.q_table[fuzzy_s1])
        td_target = reward + self.gamma * q_next_max
        self.q_table[fuzzy_s][a] = q_sa + self.alpha * (td_target - q_sa)

        self.state = next_state
        self.acc_reward += reward

        return done