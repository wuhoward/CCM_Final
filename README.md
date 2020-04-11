# CCM_Final
This code is modified from https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch. Specifically, the major changes are made in src/env.py. Please let me know if there is any bug. Thanks!

### Installing the PLE
1. git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
2. Copy the games from [here](https://github.com/rach0012/humanRL_prior_games/tree/master/ple/games) into PLE's games folder.
3. cd PyGame-Learning-Environment/
4. pip install -e .

### Notes:
* I found that training speed on CPU is on par with GPU. Probably because of larger CPU memory and the on-policy training mechanism. CIMS's [crunchy servers](https://cims.nyu.edu/webapps/content/systems/resources/computeservers) seems to be good choices
* The initial position of the player can be changed by modifying self.playerPosition in board.py. However, I haven't figured out how to map coordinates in map.txt to the real position. So I currently just modify the the values and output the screen to see the effects.
* The ICM paper states that A3C-ICM takes ~7M training steps to converge while training on VizDoom. So, I estimate our task will need ~18 hours.
* The entropy of the ICM loss can be deemed as the signal of the training progress. The current ICM loss seems slightly too large. We could adjust the value of lambda to balance the losses.
* The currently environment use a sparse reward (1 for winning). We could try to play with different settings.
* The ICM paper resizes the images to 42×42, whereas our current implementation uses 168×168. Reducing the sizes or even changing the network architecutre might speedup the convergence.
* Previous Atari papers use 4 for [frame_skip](https://pygame-learning-environment.readthedocs.io/en/latest/modules/ple.html). I noticed that frame_skip > 1 might prevent environment from reaching the game_over state sometimes.
