from stable_baselines3.common import env_checker
from environment import VizDoomGym

env = VizDoomGym(render=True)

env_checker.check_env(env)
