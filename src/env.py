import cv2
import numpy as np
import subprocess as sp
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['SDL_AUDIODRIVER'] = "dsp"
from ple.games.originalgame import originalGame
from ple import PLE
import moviepy.editor as mpy


def process_frame(frame):
    # Convert to a gray-scale image
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (42, 42))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 42, 42))

class MonsterKongEnv(object):
    def __init__(self, index, opt, output_path = None):
        self.game = originalGame()
        self.env = PLE(self.game, fps=30, display_screen=False, 
                       rng=np.random.RandomState(index), frame_skip=opt.frame_skip) # one map per game for now, no randomness
        self.train_frames = []
        self.record_frames = []
        self.reset(False, False, True)
        self.output_path = output_path

    def step(self, action):
        reward = self.env.act(self.env.getActionSet()[action])
        if self.output_path:
            self.record_frames.append(self.env.getScreenRGB())
        self.train_frames.append(process_frame(self.env.getScreenRGB()))

        game_done = self.env.game_over()
        if not game_done:
            frames = np.concatenate([frame for frame in self.train_frames[-3:]], 0)[None, :, :, :].astype(np.float32)
        else:
            frames = np.zeros((1, 3, 42, 42), dtype=np.float32)

        return frames, reward, False, False, game_done

    def reset(self, round_done, stage_done, game_done):
        self.env.reset_game()
        self.env.act(None) # dummy action to avoid black screen
        self.train_frames = [process_frame(self.env.getScreenRGB())] * 3
        self.record_frames = [self.env.getScreenRGB()]
        return np.zeros((1, 3, 42, 42), dtype=np.float32)

    def make_anim(self, fps=10, true_image=True):
        # create a video from a list of images
        images = self.record_frames
        duration = len(images) / fps
    
        def make_frame(t):
            try:
                x = images[int(len(images) / duration * t)]
            except:
                x = images[-1]
    
            if true_image:
                return x.astype(np.uint8)
            else:
                return ((x + 1) / 2 * 255).astype(np.uint8)
    
        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.fps = fps
        clip.rotate(-90).write_videofile(self.output_path, fps=fps)
    
def create_train_env(index, opt, output_path=None):
    num_inputs = 3
    num_actions = 6
    env = MonsterKongEnv(index, opt, output_path)
    return env, num_inputs, num_actions
