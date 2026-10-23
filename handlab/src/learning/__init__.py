"""Learning a general tracker: one policy that moves the object along any
motion it is given. A new demonstration needs no retraining - only its
motion, and a grip where it starts.

    trajectories.py  the library: starts the hand can grip, generated and
                     recorded motions, each tried by the planner
                     (scripts/build_library.py -> data/learning/library.npz)
    env.py           TrackEnv: the task sim, thousands at once, for rsl_rl
    train.py         training methods (PPO so far) on this Mac's cores;
                     evaluate a trained policy
                     (scripts/train_tracker.py -> data/learning/runs/<stamp>/)
    background.py    building and training as processes of their own, with
                     a status file - what the Learning panel starts and shows

The planner (src/sim) is the data engine: it finds the grips episodes start
from, and keeps only motions this hand can follow. The policy sees what the
real hand will have - encoders and a camera's estimate of the object - plus
where the object should be over the next second.
"""
