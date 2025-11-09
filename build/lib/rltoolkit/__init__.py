"""
RLToolkit - A Simple Reinforcement Learning Library

A comprehensive toolkit for implementing and experimenting with 
reinforcement learning algorithms.
"""

__version__ = "0.1.0"
__author__ = "RL Toolkit Team"
__email__ = "rltoolkit@example.com"

from .algorithms import QLearning, SARSAAgent, DQNAgent
from .environments import SimpleGridWorld, SimpleMaze
from .policies import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy
from .utils import plot_learning_curve, evaluate_agent, run_episode

__all__ = [
    'QLearning',
    'SARSAAgent', 
    'DQNAgent',
    'SimpleGridWorld',
    'SimpleMaze',
    'EpsilonGreedyPolicy',
    'GreedyPolicy', 
    'RandomPolicy',
    'plot_learning_curve',
    'evaluate_agent',
    'run_episode'
]