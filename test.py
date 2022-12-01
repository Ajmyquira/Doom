# Import eval policy to test agent
import time

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from environment import VizDoomGym

# Reload model from disk
model = PPO.load('./train/train_basic/best_model_100000.zip')

# Create rendered environment
env = VizDoomGym(render=True)

# Evaluate mean reward for 10 games
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
# print(mean_reward)

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        total_reward += reward
    print('Total reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)
