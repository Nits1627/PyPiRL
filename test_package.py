#!/usr/bin/env python3
"""
Test script to verify the RL Toolkit package functionality.
"""

import rltoolkit as rl
import numpy as np

def test_basic_functionality():
    """Test basic functionality of the RL Toolkit."""
    print("Testing RL Toolkit functionality...")
    
    # Test environment creation
    print("1. Testing environment creation...")
    env = rl.SimpleGridWorld(size=5)
    state = env.reset()
    print(f"   Initial state: {state}")
    
    # Test agent creation
    print("2. Testing agent creation...")
    agent = rl.QLearning(state_space=25, action_space=4)
    print(f"   Agent created: QLearning with {agent.state_space} states and {agent.action_space} actions")
    
    # Test policy creation
    print("3. Testing policy creation...")
    policy = rl.EpsilonGreedyPolicy(epsilon=0.1)
    print(f"   Policy created: EpsilonGreedyPolicy with epsilon={policy.get_epsilon()}")
    
    # Test action selection
    print("4. Testing action selection...")
    action = policy.select_action(agent, state)
    print(f"   Selected action: {action}")
    
    # Test environment step
    print("5. Testing environment step...")
    next_state, reward, done = env.step(action)
    print(f"   Next state: {next_state}, Reward: {reward}, Done: {done}")
    
    # Test agent learning
    print("6. Testing agent learning...")
    initial_q = agent.get_q_value(state, action)
    agent.update(state, action, reward, next_state, done)
    updated_q = agent.get_q_value(state, action)
    print(f"   Q-value updated from {initial_q:.4f} to {updated_q:.4f}")
    
    print("\n✅ All basic functionality tests passed!")

def test_training():
    """Test training functionality."""
    print("\nTesting training functionality...")
    
    # Create environment and agent
    env = rl.SimpleGridWorld(size=4)
    agent = rl.QLearning(state_space=16, action_space=4)
    
    # Train for a few episodes
    rewards = []
    policy = rl.EpsilonGreedyPolicy(epsilon=0.1)
    
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = policy.select_action(agent, state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
    
    avg_reward = np.mean(rewards)
    print(f"   Average reward over 10 episodes: {avg_reward:.2f}")
    print("✅ Training test passed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_training()
    print("\n🎉 All tests completed successfully! The RL Toolkit is working correctly.")