# CCM_Final
This code is modified from https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch. Specifically, the major changes are made in src/env.py. Please install [PLE](https://pygame-learning-environment.readthedocs.io/) and the dependencies listed in requirements-dev.txt. Let me know if there is any bug. Thanks!

### TL;DR 
```bash
# default on CPUs
python train.py
python test.py
# on GPUs
python train.py --num_processes 4 --exp my_exp --use_gpu
python test.py --exp my_exp
```

### Notes:
* I found that training speed on CPU is on par with GPU. Probably because of larger CPU memory and the on-policy training mechanism. CIMS's [crunchy servers](https://cims.nyu.edu/webapps/content/systems/resources/computeservers) seems to be good choices
* The initial position of the player can be changed by modifying self.playerPosition in board.py. However, I haven't figured out how to map coordinates in map.txt to the real position. So I currently just modify the the values and output the screen to see the effects.
* The currently game environment use a sparse reward (1 for winning). We could try to play with different settings.
