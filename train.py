# Import ppo for training
from stable_baselines3 import PPO
from environment import VizDoomGym
from test03 import LOG_DIR, callback

# Non rendered environment
env = VizDoomGym()

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
model.learn(total_timesteps=100000, callback=callback)
