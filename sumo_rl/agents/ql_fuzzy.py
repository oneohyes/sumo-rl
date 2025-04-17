"""Fuzzy Q-learning Agent class(legacy)"""
import numpy as np
import skfuzzy as fuzz
import random

from skfuzzy import control as ctrl
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class FuzzyQLearningAgent:
    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space    
        self.action = None
        self.alpha = alpha               
        self.gamma = gamma               
        self.exploration = exploration_strategy
        self.Q_table = {self.state: [0 for _ in range(self.action_space.n)]}  # using gym Discrete
        self.acc_reward = 0 
        # Initialize fuzzy control system
        self._initialize_fuzzy_system()
    
    def _initialize_fuzzy_system(self):
        """Initialize fuzzy control system."""
        self.fuzzy_states = ctrl.Antecedent(np.linspace(self.state_space.min, self.state_space.max, 100), 'state')
        self.fuzzy_actions = ctrl.Consequent(np.linspace(self.action_space.min, self.action_space.max, 100), 'action')

        self.fuzzy_states.automf(7)
        self.fuzzy_actions.automf(7)
        

    def get_fuzzy_state(self, s):
        """Get fuzzy state based on state value."""
        memberships = [fuzz.interp_membership(self.fuzzy_states.universe, self.fuzzy_states[label].mf, s) for label in self.fuzzy_states]
        return np.argmax(memberships)
    
    def act(self, state):
        """Select action based on epsilon-greedy strategy."""
        fuzzy_state = self.get_fuzzy_state(state)
        action = self.exploration.choose(self.Q_table, fuzzy_state, self.action_space)
        return action
    
    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        fuzzy_state = self.get_fuzzy_state(self.state)
        fuzzy_next_state = self.get_fuzzy_state(next_state)
        action = self.act(next_state)
        
        # Q-learning 更新公式
        best_next_q = np.max(self.Q_table[fuzzy_next_state])
        td_target = reward + self.gamma * best_next_q
        self.Q_table[fuzzy_state][action] += self.alpha * (td_target - self.Q_table[fuzzy_state][action])
        self.state = next_state
        self.acc_reward += reward
    
    def print_Q_table(self):
        """打印学习后的 Q 表"""
        print("\nLearned Fuzzy Q-table:")
        for i, label in enumerate(self.fuzzy_states):
            print(f"State '{label}':", self.Q_table[i])
