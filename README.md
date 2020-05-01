# CCM_Final
This code is modified from https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch. Specifically, the major changes are made in src/env.py. Please install [PLE](https://pygame-learning-environment.readthedocs.io/) and the dependencies listed in requirements-dev.txt. Let me know if there is any bug. Thanks!

### TL;DR 
```bash
# running on CPUs
python train.py --num_processes 16 --map_file fire_3.txt --exp my_exp
python test.py --resume_path trained_models/my_exp/500K --map_file fire_3.txt
# visualize with tensorboard
tensorboard --logdir tensorboard --bind_all &
```

### Notes:
* I found that training speed on CPU is on par with GPU. Probably because of larger CPU memory and the on-policy training mechanism. CIMS's [crunchy servers](https://cims.nyu.edu/webapps/content/systems/resources/computeservers) seems to be good choices.
* Not recommend using to many processes, the A3C paper uses 16 processes.
* --max_steps needs to be greater than 500 for longer exploration
* Decreasing the learning rate and changing the weight (--lambda) for curiosity loss might be worth tuning.
* The ICM paper uses --lr=1e-3. However, from my experiments it can't be that large.
* Previous papers use --frame_skip=4.
* Removing the null action (or set --num_action=5) doesn't help.
* Adding negative rewards doesn't work on map.txt. It's worth looking into how to change the hyperparameters to make it work.
* The initial position of the player can be changed by modifying self.playerPosition in board.py. However, I haven't figured out how to map coordinates in map.txt to the real position. So I currently just modify the the values and output the screen to see the effects.

### Known Bugs:
* Tensorboard cannot show results from every process, probably due to race condition?
