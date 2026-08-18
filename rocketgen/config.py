"""Shared data contracts for the nTop + SUAVE rocket generator.

Every work package imports from here. Do not redefine these shapes locally.

Units are SI throughout: metre, kilogram, second, radian, newton, pascal, kelvin.
Convert at the boundary only (nTop literals are metres and radians, so no conversion
is needed there; report tables convert for display).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# --------------------------------------------------------------------------------------
#   Paths
# --------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(REPO_ROOT, "vendor")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

FUNCTIONS_JSON = os.path.join(VENDOR_DIR, "functions.json")
TYPES_JSON = os.path.join(VENDOR_DIR, "types.json")
TYPE_DEFAULTS_JSON = os.path.join(VENDOR_DIR, "type_defaults.json")

# Path to the nTop Automate command-line tool.
#
# Set the NTOPCL environment variable to point at your own install or build. The default is the
# standard Windows install location. The block universe in `vendor/` must come from the same nTop
# version, or a block signature may not resolve; `scripts/bootstrap.py --check` reports both.
NTOPCL_PATH = os.environ.get("NTOPCL", r"C:/Program Files/nTopology/nTopology/ntopcl.exe")
NTOPCL_FALLBACK = os.environ.get(
    "NTOPCL_FALLBACK", r"C:/Program Files/nTop/nTop/ntopcl.exe"
)


def add_suave_to_path() -> None:
    """Put the vendored SUAVE 2.5.2 on sys.path. Idempotent."""
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)


# --------------------------------------------------------------------------------------
#   Design vector
# --------------------------------------------------------------------------------------


@dataclass
class DesignVector:
    """The geometry and propulsion parameters the sizer moves.

    These are exactly the values handed to the nTop notebook as notebook inputs, plus the
    propulsion values the notebook needs to cut the grain cavity.

    Bounds are in SPEC.md section 3. `bounds()` returns them so the optimiser and the DOE
    driver share one definition.
    """

    # --- body ---
    D: float = 0.35                  # body diameter, m
    L_total: float = 4.00            # overall length, nose tip to nozzle exit plane, m
    f_nose: float = 3.0              # nose fineness, L_nose / D, dimensionless
    nose_shape: str = "tangent_ogive"  # "tangent_ogive" | "cone" (cone is for validation cases)
    t_wall: float = 0.0030           # body wall thickness, m
    L_boattail: float = 0.20         # aft boattail length, m
    d_base: float = 0.30             # base (nozzle exit plane) diameter, m

    # --- splined outer mould line ---
    #
    # `nose_shape = "spline"` swaps the one-parameter tangent-ogive family for a clamped cubic
    # B-spline. `nose_blend` then selects a shape along a ONE-DIMENSIONAL family:
    #
    #     0.0  the ogive-equivalent spline, which reproduces the tangent ogive to 1e-6 of R
    #          and gives a wave-drag shape ratio of exactly 1.0
    #     1.0  the spline that minimises slender-body wave drag at this control-point count
    #
    # One scalar suffices because the drag-optimal NORMALISED profile is fineness-invariant
    # (measured to 6.2e-9), so the shape sub-problem separates from sizing. Six free control
    # values would otherwise enter the design vector and the DOE for no extra reach.
    #
    # The trade this scalar buys, measured at f_nose 3.4: nose wave drag 1.000 -> 0.875, nose
    # enclosed volume -6.35 percent, nose wetted area -2.7 percent. Less drag, less room for
    # the seeker, and a centre of pressure that moves aft.
    nose_blend: float = 0.0
    n_ctrl_oml: int = 9              # spline control points; see oml_spline.SOURCES

    # Boattail: `boattail_shape = "spline"` replaces the straight cone with a splined
    # contraction. `boattail_blend` runs 0.0 (straight cone, exactly the old geometry) to 1.0
    # (maximum curvature allowed by the control polygon).
    boattail_shape: str = "cone"     # "cone" | "spline"
    boattail_blend: float = 0.0

    # --- fins: cruciform, 4 panels ---
    n_fin: int = 4
    b_fin: float = 0.18              # exposed semi-span per panel, m
    c_r_fin: float = 0.42            # root chord, m
    taper_fin: float = 0.45          # tip chord / root chord
    sweep_fin: float = math.radians(45.0)   # leading-edge sweep, rad
    t_fin: float = 0.012             # fin maximum thickness, m
    x_fin_te_gap: float = 0.05       # gap from fin trailing edge to base, m

    # --- propulsion ---
    m_p_boost: float = 100.0         # boost propellant mass, kg
    m_p_sustain: float = 260.0       # sustain propellant mass, kg
    F_boost: float = 45.0e3          # boost thrust, N (vacuum-corrected inside the motor model)

    # Terminal boost. Ignited at dive entry, not on a timer.
    #
    # This phase exists because SPEC R6 (Mach 1.50 at sea-level impact) cannot be met by an
    # unpowered dive at any dive angle. The dive is terminal-velocity limited: at the SV-1
    # burnout mass and calibrated CD, sqrt(2*m*g/(rho*S*CD)) at sea level is 315 m/s, which is
    # Mach 0.93. Sweeping the dive angle from -25 to -89 degrees moves impact Mach only from
    # 0.66 to 0.97. Thrust in the endgame is the only physical route to R6.
    #
    # Terminal propellant is taken OUT of the total propellant budget, not added to it.
    m_p_terminal: float = 0.0        # terminal-boost propellant mass, kg
    F_terminal: float = 0.0          # terminal-boost thrust, N
    eps_nozzle: float = 8.0          # nozzle area ratio, Ae/At
    p_c: float = 7.0e6               # chamber pressure, Pa

    # --- internal layout, stations measured from the nose tip, m ---
    L_seeker: float = 0.30           # seeker and radome bay length
    L_guidance: float = 0.25         # guidance and actuation bay length
    L_warhead: float = 0.55          # warhead bay length

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Optimiser bounds. SPEC.md section 3."""
        return {
            "D": (0.25, 0.45),
            "L_total": (3.00, 4.20),
            "f_nose": (2.0, 4.0),
            "m_p_boost": (40.0, 250.0),
            "m_p_sustain": (100.0, 500.0),
            "F_boost": (20.0e3, 90.0e3),
            "b_fin": (0.10, 0.30),
            "c_r_fin": (0.25, 0.60),
            "taper_fin": (0.20, 0.80),
            "sweep_fin": (math.radians(20.0), math.radians(60.0)),
            # Terminal boost. Both bounds are derived, not picked.
            #
            # m_p_terminal upper: the terminal grain is a tube, so its web is V_p / A_b and the
            # web must stay inside the bay radius. At the 338 mm default bore and dv.p_c the
            # burning area is 0.190 m^2, giving 0.190 * 0.169 * 1800 = 57.8 kg. 60 leaves
            # headroom for a larger bore.
            #
            # F_terminal lower: below about 2 kN the terminal chamber pressure falls under 1 MPa,
            # where composite propellant combustion is unreliable. Upper: 15 kN needs about
            # 12 MPa, which thickens the motor case from 1.20 to 1.72 mm and adds 3 kg; beyond
            # that the case penalty eats the benefit.
            "m_p_terminal": (0.0, 60.0),
            "F_terminal": (2.0e3, 15.0e3),
            # Shape blends. Both are pure interpolation parameters between two shapes that
            # are each known to be valid, so the bounds are the definition of the family and
            # not a judgement call. Outside [0, 1] the spline extrapolates and stops being
            # monotone, which `oml_spline.SplineProfile.is_monotone` would catch.
            "nose_blend": (0.0, 1.0),
            "boattail_blend": (0.0, 1.0),
        }

    # --- derived geometry, single source of truth ---

    @property
    def L_nose(self) -> float:
        """Nose length, m."""
        return self.f_nose * self.D

    @property
    def is_splined(self) -> bool:
        """True when the outer mould line uses the spline family anywhere."""
        return self.nose_shape == "spline" or self.boattail_shape == "spline"

    @property
    def nose_control(self) -> tuple[float, ...] | None:
        """Spline control values of the nose, or None when the nose is not splined.

        SINGLE SOURCE OF TRUTH. The nTop notebook, the mass build-up, the aero build-up and
        the wave-drag model all read the shape from here, so they cannot drift apart.

        The blend is linear in the control values, and the radius is linear in the control
        values, so the blend is linear in the profile too. That keeps every blend a valid
        monotone nose (verified across the range) rather than only the two endpoints.
        """
        if self.nose_shape != "spline":
            return None
        from .oml_spline import ogive_control_values
        from .sizing.wavedrag import optimal_control_values

        k = self.L_nose / (0.5 * self.D)
        base = ogive_control_values(k, self.n_ctrl_oml)
        best = optimal_control_values(self.n_ctrl_oml)
        b = float(self.nose_blend)
        return tuple((1.0 - b) * lo + b * hi for lo, hi in zip(base, best))

    @property
    def boattail_control(self) -> tuple[float, ...] | None:
        """Spline control values of the boattail, or None when it is a straight cone.

        The run goes from the body radius down to the base radius, so the control values are
        expressed on the CONTRACTION: 0 at the start of the run, 1 at the base. `blend = 0`
        reproduces the straight cone exactly.
        """
        if self.boattail_shape != "spline":
            return None
        from .oml_spline import boattail_control_values

        return boattail_control_values(float(self.boattail_blend), self.n_ctrl_oml)

    @property
    def S_ref(self) -> float:
        """Aerodynamic reference area: body maximum cross-section, m^2."""
        return 0.25 * math.pi * self.D ** 2

    @property
    def S_base(self) -> float:
        """Base area at the nozzle exit plane, m^2."""
        return 0.25 * math.pi * self.d_base ** 2

    @property
    def L_body_cyl(self) -> float:
        """Length of the constant-diameter cylindrical section, m."""
        return self.L_total - self.L_nose - self.L_boattail

    @property
    def c_t_fin(self) -> float:
        """Fin tip chord, m."""
        return self.taper_fin * self.c_r_fin

    @property
    def S_fin_exposed(self) -> float:
        """Exposed planform area of ONE fin panel, m^2."""
        return 0.5 * (self.c_r_fin + self.c_t_fin) * self.b_fin

    @property
    def x_fin_le(self) -> float:
        """Fin root leading-edge station from the nose tip, m."""
        return self.L_total - self.x_fin_te_gap - self.c_r_fin

    @property
    def fineness(self) -> float:
        """Overall body fineness ratio, L/D."""
        return self.L_total / self.D

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            L_nose=self.L_nose,
            S_ref=self.S_ref,
            S_base=self.S_base,
            L_body_cyl=self.L_body_cyl,
            c_t_fin=self.c_t_fin,
            S_fin_exposed=self.S_fin_exposed,
            x_fin_le=self.x_fin_le,
            fineness=self.fineness,
        )
        return d

    def replace(self, **kw: Any) -> "DesignVector":
        return replace(self, **kw)

    def geometry_is_valid(self) -> tuple[bool, list[str]]:
        """Cheap geometric sanity gate. Run before spending an nTop call."""
        errs: list[str] = []
        if self.L_body_cyl <= 0.30:
            errs.append(f"cylindrical section too short: {self.L_body_cyl:.3f} m")
        if self.d_base > self.D:
            errs.append(f"base diameter {self.d_base:.3f} exceeds body {self.D:.3f}")
        if self.x_fin_le <= self.L_nose:
            errs.append("fin leading edge falls on the nose")
        if self.t_wall <= 0.0 or self.t_wall > 0.05 * self.D:
            errs.append(f"wall thickness {self.t_wall:.4f} m out of range")
        bay = self.L_seeker + self.L_guidance + self.L_warhead
        if bay >= self.L_total - self.L_boattail:
            errs.append("forward bays do not leave room for the motor")
        return (not errs), errs


# --------------------------------------------------------------------------------------
#   Requirements
# --------------------------------------------------------------------------------------


@dataclass
class Requirements:
    """SPEC.md section 2. Demo values, not from any real programme."""

    M_launch: float = 0.85
    h_launch: float = 10_000.0        # m
    M_cruise: float = 2.00
    h_cruise: float = 12_000.0        # m
    range_min: float = 185_000.0      # m
    m_warhead: float = 90.0           # kg
    m_guidance: float = 15.0          # kg
    M_terminal_min: float = 1.50
    D_max: float = 0.45               # m
    L_max: float = 4.20               # m
    m0_max: float = 1100.0            # kg
    static_margin_min: float = 1.0    # calibres
    b_fin_span_max: float = 0.90      # m, tip to tip

    # Structural dynamic-pressure limit.
    #
    # The first value written into this spec was 90 kPa. A requirements audit showed that it is
    # mutually exclusive with R6. R6 demands Mach 1.50 at impact; at sea level that is 510.4 m/s,
    # which is 159.6 kPa. A 90 kPa limit caps sea-level impact at Mach 1.13, so no design can
    # satisfy both. The limit, not the physics, was wrong.
    #
    # 200 kPa is used instead. It clears the 159.6 kPa floor that R6 itself sets, with 25 percent
    # margin, and it is consistent with the manoeuvre-envelope pressures quoted for supersonic
    # tactical missiles in Fleeman, Tactical Missile Design, 2nd ed., Chapter 3, where sea-level
    # supersonic flight at 150 to 250 kPa is normal for this class.
    q_max: float = 200_000.0          # Pa, structural limit. See the note above.
    t_separation: float = 1.5         # s of unpowered separation before boost
    gamma_terminal: float = math.radians(-35.0)   # terminal dive flight-path angle, rad


# --------------------------------------------------------------------------------------
#   Materials
# --------------------------------------------------------------------------------------


@dataclass
class Material:
    name: str
    density: float          # kg/m^3
    sigma_yield: float      # Pa
    E: float                # Pa
    source: str


# Densities and strengths are handbook values for the named alloys and composites.
# Sources are given per entry. These are the only material numbers in the codebase.
MATERIALS: dict[str, Material] = {
    "airframe_al7075": Material(
        name="Aluminium 7075-T6",
        density=2810.0,
        sigma_yield=503e6,
        E=71.7e9,
        source="ASM Aerospace Specification Metals, 7075-T6 datasheet",
    ),
    "motorcase_4130": Material(
        name="Steel 4130 normalised",
        density=7850.0,
        sigma_yield=460e6,
        E=205e9,
        source="ASM Handbook Vol.1, AISI 4130 normalised at 870 C",
    ),
    "motorcase_cfrp": Material(
        name="Carbon/epoxy filament wound",
        density=1550.0,
        sigma_yield=1500e6,
        E=135e9,
        source="MIL-HDBK-17-2F, unidirectional AS4/3501-6 hoop properties",
    ),
    "fin_ti64": Material(
        name="Titanium Ti-6Al-4V",
        density=4430.0,
        sigma_yield=880e6,
        E=113.8e9,
        source="ASM Aerospace Specification Metals, Ti-6Al-4V annealed",
    ),
    "propellant_htpb_ap": Material(
        name="HTPB/AP/Al composite solid propellant",
        density=1800.0,
        sigma_yield=0.0,
        E=0.0,
        source=(
            "Sutton and Biblarz, Rocket Propulsion Elements 9th ed., Table 12-1: "
            "AP/HTPB/Al composite density 1.77 to 1.86 g/cm^3"
        ),
    ),
    "insulation_epdm": Material(
        name="EPDM motor insulation",
        density=1100.0,
        sigma_yield=0.0,
        E=0.0,
        source="Sutton and Biblarz, Rocket Propulsion Elements 9th ed., Chapter 14, EPDM liner",
    ),
    "radome_pyroceram": Material(
        name="Pyroceram 9606 radome ceramic",
        density=2600.0,
        sigma_yield=250e6,
        E=120e9,
        source="Corning Pyroceram 9606 datasheet, as used for rocket radomes",
    ),
}


# --------------------------------------------------------------------------------------
#   Measurement contract - what nTop hands back to SUAVE
# --------------------------------------------------------------------------------------


@dataclass
class NtopMeasurements:
    """Everything the nTop notebook measures and returns. WP1/WP4 fill this; WP2/WP5 consume it.

    `None` means the notebook did not report the quantity. Consumers must handle that by
    falling back to an analytic estimate AND recording that they did so.
    """

    # solid volumes, m^3
    volume_total: float | None = None          # outer mould line enclosed volume
    volume_structure: float | None = None      # airframe walls + fins + bulkheads
    volume_cavity: float | None = None         # usable internal volume
    volume_grain: float | None = None          # propellant grain volume as modelled

    # areas, m^2
    area_wetted_body: float | None = None
    area_wetted_fins: float | None = None      # all panels, both sides
    area_base: float | None = None

    # mass properties of the structure only (density x volume inside nTop), kg and kg.m^2
    mass_structure: float | None = None
    cg_structure: tuple[float, float, float] | None = None
    inertia_structure: tuple[float, float, float] | None = None   # Ixx, Iyy, Izz about CG

    # cross-section area distribution, for wave drag: list of (x_from_nose_tip_m, area_m2)
    area_distribution: list[tuple[float, float]] = field(default_factory=list)

    # exported artefacts
    stl_path: str | None = None
    step_path: str | None = None
    implicit_path: str | None = None
    ntop_path: str | None = None

    # bookkeeping
    wall_time_s: float | None = None
    ntopcl_returncode: int | None = None
    warnings: list[str] = field(default_factory=list)

    def is_usable(self) -> bool:
        """True when the measurements needed by the aero and mass models are present."""
        needed = (
            self.volume_total,
            self.volume_cavity,
            self.area_wetted_body,
            self.mass_structure,
        )
        return all(v is not None for v in needed)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


# --------------------------------------------------------------------------------------
#   Mass statement
# --------------------------------------------------------------------------------------


@dataclass
class MassStatement:
    """Group-weight statement. Every entry carries its own moment arm from the nose tip."""

    items: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (mass kg, x m)

    def add(self, name: str, mass: float, x_cg: float) -> None:
        self.items[name] = (mass, x_cg)

    @property
    def total_mass(self) -> float:
        return sum(m for m, _ in self.items.values())

    @property
    def x_cg(self) -> float:
        m_tot = self.total_mass
        if m_tot <= 0.0:
            return 0.0
        return sum(m * x for m, x in self.items.values()) / m_tot

    def without(self, *names: str) -> "MassStatement":
        """A copy with the named items removed. Use for burnout mass."""
        keep = {k: v for k, v in self.items.items() if k not in names}
        return MassStatement(items=keep)


# --------------------------------------------------------------------------------------
#   Aero and trajectory contracts
# --------------------------------------------------------------------------------------


@dataclass
class AeroCoefficients:
    """Aerodynamic state at one (Mach, altitude, alpha) point."""

    mach: float
    altitude: float
    alpha: float
    CD0: float                  # zero-lift drag, on S_ref
    CD: float                   # total drag, on S_ref
    CN: float                   # normal force, on S_ref
    CN_alpha: float             # per radian
    CM: float                   # pitching moment about the nose tip, on S_ref * D
    x_cp: float                 # centre of pressure from the nose tip, m
    L_over_D: float
    breakdown: dict[str, float] = field(default_factory=dict)   # named drag contributions


@dataclass
class TrajectoryResult:
    """Output of the 3-DOF integration."""

    time: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)          # ground range, m
    h: list[float] = field(default_factory=list)          # altitude, m
    V: list[float] = field(default_factory=list)          # true airspeed, m/s
    mach: list[float] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)
    gamma: list[float] = field(default_factory=list)
    thrust: list[float] = field(default_factory=list)
    drag: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)          # dynamic pressure, Pa
    alpha: list[float] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)

    # Fin-authority diagnostics. `CN_required` is the normal-force coefficient the guidance
    # commanded; `alpha_limited` flags steps where the alpha limit clipped the command, which
    # means the fins could not deliver the commanded manoeuvre.
    CN_required: list[float] = field(default_factory=list)
    alpha_limited: list[bool] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    converged: bool = False
    message: str = ""

    @property
    def range_final(self) -> float:
        return self.x[-1] if self.x else 0.0

    @property
    def mach_final(self) -> float:
        return self.mach[-1] if self.mach else 0.0

    @property
    def q_max(self) -> float:
        return max(self.q) if self.q else 0.0


# --------------------------------------------------------------------------------------
#   Source registry
# --------------------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    "US_Standard_1976": "NASA-TM-X-74335, U.S. Standard Atmosphere 1976; via SUAVE 2.5.2",
    "SUAVE": "Stanford ADL, SUAVE 2.5.2 (Mar 2022), github.com/suavecode/SUAVE",
    "nTop": "nTop 5.53.2 / 5.54.0, ntopcl Automate",
    "cd0_calibration": (
        "Calibration, not physics. The WP2 component build-up reproduces the 23 Basic Finner "
        "free-flight shots of Dupuis and Hathaway, DREV-TM-9703 (1997) Table VII with a "
        "systematic mean bias of -14.6 percent on CD0 for M >= 1.4. The shortfall is "
        "attributed to fin trailing-edge base drag, fin-body junction interference, and the "
        "double-wedge fin section assumed in place of the real conical section. CD0 is scaled "
        "by 1/(1 - 0.146) = 1.171 in the sizing loop so that range is not overpredicted. "
        "Applied once, at the loop boundary, never inside the aero model."
    ),
}

# Multiplicative correction applied to CD0 in the sizing loop. See SOURCES["cd0_calibration"].
CD0_CALIBRATION = 1.171


def register_sources(new: dict[str, str]) -> None:
    """Modules call this at import to declare where their constants come from."""
    for k, v in new.items():
        if k in SOURCES and SOURCES[k] != v:
            raise ValueError(f"conflicting source for {k!r}: {SOURCES[k]!r} vs {v!r}")
        SOURCES[k] = v
