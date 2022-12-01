from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import numpy as np
import time


class VizDoomGym(Env):
    def __init__(self, render=False):
        # Inherit from Env
        super().__init__()

        # Set up the game
        self.game = DoomGame()
        self.game.load_config('Github/VizDoom/scenarios/basic.cfg')
        self.game.init()

        # Render frame logic
        if not render:
            self.game.set_window_visible(False)
            # print("Not rendering")
        else:
            self.game.set_window_visible(True)
            # print("Rendering")

        # Start the game
        self.game.init()

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        # Get all the other stuff we need to return
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

    def render(self):
        pass

    # What happens when start a new game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    # Grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    # Call to close down the game
    def close(self):
        self.game.close()


# env = VizDoomGym()
# env.reset()
# for i in range(10):
#     time.sleep(0.2)
#     print(env.step(2))
# env.reset()
# env.close()
