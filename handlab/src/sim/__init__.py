"""Trying a task in simulation before the real hand.

    task.py     the task sim: the robot with position servos, a free object,
                a distance sensor from each fingertip to it
    planner.py  a grip (cross-entropy search over servo targets), then
                predictive sampling: many short finger motions tried in
                parallel each step, the best one used

processing/plan.py runs them on a recording (Recordings, step 7).
"""
