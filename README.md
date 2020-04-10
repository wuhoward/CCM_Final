# CCM_Final
This code in modified from https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch. Specifically, the major changes are made in src/env.py. Please let me know if there is any bug. Thanks!

### Installing the PLE
1. Follow the instruction from https://github.com/rach0012/humanRL_prior_games#installation
2. Copy the games from [here](https://github.com/rach0012/humanRL_prior_games/tree/master/ple/games) into PLE's games folder.
3. Install

### Notes:
* I found that training speed on CPU is on par with GPU. Probably because of larger CPU memory and the on-policy training mechanism.
* The initial position of the player can be changed by modifying self.playerPosition in board.py. However, I haven't figured out how to map coordinates in map.txt to the real position. So I currently just modify the the values and output the screen to see the effects.
* The original paper of ICM stated that it takes A3C-ICM 4M steps to converge while training on VizDoom. So, I estimate it will takes more than 12 hours
* The current ICM loss dosen't change much during training. This my be a potential bug?
* Previous Atari paper seems to use a value in \[3, 5\] for [frame_skip](https://pygame-learning-environment.readthedocs.io/en/latest/modules/ple.html). I'm still trying to figure out what's the proper setting.
