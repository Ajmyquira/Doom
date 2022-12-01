from vizdoom import *
import random
import time
# Import numpy for identity matrix
import numpy as np

# Setup game
game = DoomGame()
game.load_config('Github/VizDoom/scenarios/basic.cfg')
game.init()

# This is the set of actions we can take in the environment
actions = np.identity(3, dtype=np.uint8)

# Loop through episodes
episodes = 10
for episode in range(episodes):
    # Create a new episode or game
    game.new_episode()
    # Check the game isn't done
    while not game.is_episode_finished():
        # Get the game state
        state = game.get_state()
        # Get the game image
        img = state.screen_buffer
        # Get the game variable (ammo)
        info = state.game_variables
        # Take the action
        reward = game.make_action(random.choice(actions), 4)
        print('Reward:', reward)
        time.sleep(0.02)
    print('Result:', game.get_total_reward())
    time.sleep(2)

game.close()
