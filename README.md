# GYM-AO
`gym-ao` is a simulation framework based on Gymnasium for testing Reinforcement Learning (RL) algorithms for Adaptive Optics (AO), specifically for focal plane wavefront control.
Two environments have been developed:
- `gym_sharpening.py`: Focal plane wavefront control with the goal of maximizing the Strehl ratio based on focal plane images.
- `gym_darkhole.py`: Dark hole digging based on pairwise probing with the goal of maximizing contrast.

## Example usage:
```
import gym_ao

def run():
    env = Sharpening_AO_system()
    N_iter = 100
    N_episode = 10
    for episode in range(N_episode):
        o = env.reset()
        print('Episode:', env.episode)
        for i in range(N_iter):
            a = 0.1 * env.action_space.sample()
            o, r, t, trunc, info = env.step(a)
            if trunc:
                break
            env.render()
    env.close()
```
