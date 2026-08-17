"""A deliberately trivial aerodynamic model, for testing the WP3 trajectory integrator.

This is NOT a physics model and it must never be imported by `rocketgen/`. WP2 owns the
real build-up (`rocketgen/sizing/aero.py`, class `RocketAero`). This stub exists only so
the WP3 tests can run before WP2 lands, and so the analytic validation cases can switch
drag on and off at will.

    CD = CD0 + k * CN^2
    CN = CN_alpha * alpha

Every coefficient is a made-up constant. That is the point: the WP3 tests validate the
integrator, not the aerodynamics.
"""
from __future__ import annotations

import math

from rocketgen.config import AeroCoefficients


class SimpleAero:
    """Constant-CD0, linear-CN aerodynamic stub matching the `AeroCallable` protocol."""

    def __init__(
        self,
        CD0: float = 0.40,
        CN_alpha: float = 12.0,
        induced_factor: float = 0.35,
        x_cp: float = 2.4,
        reference_length: float = 0.35,
        base_drag_relief: float = 0.06,
    ) -> None:
        self.CD0 = CD0
        self.CN_alpha = CN_alpha
        self.induced_factor = induced_factor
        self._x_cp = x_cp
        self.reference_length = reference_length
        self.base_drag_relief = base_drag_relief

    def evaluate(
        self, mach: float, altitude: float, alpha: float, power_on: bool = False
    ) -> AeroCoefficients:
        cn = self.CN_alpha * alpha
        cd0 = self.CD0 - (self.base_drag_relief if power_on else 0.0)
        cd = cd0 + self.induced_factor * cn * cn
        cm = -cn * self._x_cp / self.reference_length
        lift = cn * math.cos(alpha) - cd * math.sin(alpha)
        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=alpha,
            CD0=cd0,
            CD=cd,
            CN=cn,
            CN_alpha=self.CN_alpha,
            CM=cm,
            x_cp=self._x_cp,
            L_over_D=lift / cd if cd > 0.0 else 0.0,
            breakdown={"CD0": cd0, "CD_induced": self.induced_factor * cn * cn},
        )

    def trim_alpha(self, mach: float, altitude: float, required_CN: float) -> float:
        return required_CN / self.CN_alpha


class ConstantDragAero:
    """Fixed CD, zero normal force. Used for the terminal-velocity analytic case."""

    def __init__(self, CD: float = 0.5) -> None:
        self.CD = CD

    def evaluate(
        self, mach: float, altitude: float, alpha: float, power_on: bool = False
    ) -> AeroCoefficients:
        return AeroCoefficients(
            mach=mach,
            altitude=altitude,
            alpha=0.0,
            CD0=self.CD,
            CD=self.CD,
            CN=0.0,
            CN_alpha=0.0,
            CM=0.0,
            x_cp=0.0,
            L_over_D=0.0,
        )

    def trim_alpha(self, mach: float, altitude: float, required_CN: float) -> float:
        return 0.0
