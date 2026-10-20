"""Temperature protection for the servos.

A gripping hand sits at its current limit, and a stalled XL330 turns all of
that current into heat. The servo's own guard is its EEPROM temperature limit
(70 deg C by default): it shuts the motor off and latches an error that needs
a reboot. This acts earlier, and gradually:

    below temp_derate_c          full current
    temp_derate_c..temp_cutoff_c that joint's current cap falls linearly
                                 towards zero - the finger gives way under
                                 load instead of cooking
    at temp_cutoff_c             torque off for the whole hand, sending stops,
                                 and arming is refused until every joint is
                                 COOL_MARGIN below the cutoff

The cap is the goal-current register, a RAM register that bounds the position
loop's output in mode 5, so derating needs no torque-off. It is written in
STEP increments, so a temperature hovering on a boundary does not rewrite the
register every frame.
"""

from __future__ import annotations

import math

import numpy as np

STEP = 0.1               # derating moves in 10 % steps of the current limit
COOL_MARGIN = 5.0        # deg C below the cutoff before arming is allowed again


class ThermalGuard:
    def __init__(self, joint_names, derate_c, cutoff_c):
        self.joint_names = list(joint_names)
        self.derate_c = float(derate_c)
        self.cutoff_c = float(cutoff_c)
        self.temperature = np.full(len(self.joint_names), np.nan)
        self.scale = np.ones(len(self.joint_names))     # of the current limit
        self.tripped = False

    def update(self, temperature):
        """Take a frame of temperatures (NaN = not read this frame).

        Returns (changed, trip): `changed` lists (joint index, new scale) for
        caps that moved a step; `trip` is True on the frame the cutoff is hit.
        """
        temperature = np.asarray(temperature, dtype=float)
        fresh = ~np.isnan(temperature)
        self.temperature[fresh] = temperature[fresh]

        changed = []
        for index, celsius in enumerate(self.temperature):
            if math.isnan(celsius):
                continue
            scale = self.scale_for(celsius)
            if scale != self.scale[index]:
                self.scale[index] = scale
                changed.append((index, scale))

        trip = not self.tripped and bool(np.nanmax(self.temperature, initial=-np.inf)
                                         >= self.cutoff_c)
        if trip:
            self.tripped = True
        elif self.tripped and self.cool():
            self.tripped = False
        return changed, trip

    def scale_for(self, celsius):
        if celsius >= self.cutoff_c:
            return 0.0
        if celsius <= self.derate_c or self.derate_c >= self.cutoff_c:
            return 1.0
        left = (self.cutoff_c - celsius) / (self.cutoff_c - self.derate_c)
        return round(max(STEP, math.floor(left / STEP) * STEP), 2)

    def cool(self):
        known = self.temperature[~np.isnan(self.temperature)]
        return bool(known.size == 0 or known.max() < self.cutoff_c - COOL_MARGIN)

    def hottest(self):
        """(joint name, deg C) of the hottest joint read so far, or None."""
        if np.all(np.isnan(self.temperature)):
            return None
        index = int(np.nanargmax(self.temperature))
        return self.joint_names[index], float(self.temperature[index])

    def status(self):
        hot = self.hottest()
        derated = [self.joint_names[i] for i in np.flatnonzero(self.scale < 1.0)]
        level = "trip" if self.tripped else ("derate" if derated else "ok")
        return {"level": level, "hottest": hot, "derated": derated,
                "temperatures": dict(zip(self.joint_names,
                                         [None if math.isnan(t) else round(float(t), 1)
                                          for t in self.temperature]))}
