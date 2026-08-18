"""Per-stage group-weight build-up for IV-1. SPEC_IV1.md section 7.

Two things make this different from the single-body `masses.py`:

1. **Mass leaves the vehicle.** The booster and the interstage are jettisoned at separation, so the
   statement has to be per stage AND the jettisoned total has to be exact. The trajectory subtracts
   it in one step, so an error here shows up directly as a velocity error after staging.
2. **The payload stage carries the payload.** Its own inert mass, its own propellant, and the
   payload that must survive to intercept.

`masses.py` remains the authority for the shared physics: the hoop-stress case sizing, the
minimum-gauge floor, the tactical-motor correlation band and the ogive quadrature all come from
there rather than being written again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import MATERIALS, NtopMeasurements, register_sources
from ..config_iv1 import (
    InterceptRequirements,
    StackDesignVector,
    StageSpec,
    StrakeSpec,
)
from .masses import (
    BULKHEAD_T_FRAC,
    DRY_MARGIN,
    N_BULKHEAD,
    T_INSULATION,
    T_RADOME,
    MassBuildup,
    _tangent_ogive_surface_area,
    _tangent_ogive_volume,
    motor_case_mass,
)

SOURCES: dict[str, str] = {
    "iv1_stage_mass_split": (
        "Per-stage group weights are built from the same correlations and physics as the "
        "single-body build-up in masses.py: hoop-stress case sizing with a minimum-gauge floor, "
        "constant-thickness insulation, and the tactical-motor propellant mass fraction band of "
        "0.80 to 0.92 from Fleeman, Tactical Missile Design, 2nd ed., Chapter 4."
    ),
    "iv1_interstage_mass": (
        "MODELLING CHOICE: the interstage is a truncated conical shell between the two stage "
        "diameters, of thickness dv.t_interstage in aluminium 7075-T6, plus two ring frames at "
        "40 percent of the shell thickness. Separation hardware (explosive bolts or a linear "
        "separation joint, springs or pistons, and the electrical disconnect) is NOT costed. "
        "GUESS: that omission makes the jettisoned mass optimistic by an unquantified amount."
    ),
    "iv1_strake_mass": (
        "Strake mass is the swept plate volume times the titanium density in "
        "MATERIALS['fin_ti64'], with a 0.85 area-to-volume factor for a tapered leading edge and "
        "a root fillet. GUESS: the 0.85 factor is not from a source; a constant-section plate "
        "would be 1.0 and a full double wedge 0.5, so it sits between them."
    ),
    "iv1_interstage_buckling": (
        "The interstage is a compression member: it transmits the full booster thrust into the "
        "upper stage. Classical buckling of a thin unstiffened cylindrical shell under axial "
        "compression is sigma_cr = E*t / (R*sqrt(3*(1-nu^2))), which is 0.605*E*t/R at nu = 0.33. "
        "Real shells reach only a fraction of that because of imperfection sensitivity. A "
        "knockdown factor of 0.30 is applied, which is the conservative end of the 0.15 to 0.50 "
        "band quoted for unstiffened monocoque shells in NASA SP-8007, Buckling of Thin-Walled "
        "Circular Cylinders. GUESS: 0.30 was chosen inside that band rather than computed from "
        "an imperfection survey, and the shell here is conical rather than cylindrical, so the "
        "check is an approximation applied at the smaller radius, which is the critical end."
    ),
    "iv1_payload_density": (
        "Payload packaging density of 1500 kg/m^3 for the volume-closure check on the payload "
        "stage. Fleeman, Tactical Missile Design, 2nd ed., Chapter 2 gives 800 to 1600 kg/m^3 "
        "for missile electronics and actuation packaging; the upper part of that band is taken "
        "because a payload stage is denser than an avionics bay."
    ),
}
register_sources(SOURCES)

STRAKE_AREA_TO_VOLUME = 0.85     # SOURCES["iv1_strake_mass"] - a guess
RHO_PAYLOAD = 1500.0             # kg/m^3. SOURCES["iv1_payload_density"]
N_RING_INTERSTAGE = 2            # SOURCES["iv1_interstage_mass"]
BUCKLING_KNOCKDOWN = 0.30        # SOURCES["iv1_interstage_buckling"]
BUCKLING_SF = 1.25               # design safety factor on the buckling allowable


def interstage_buckling_check(dv: StackDesignVector) -> dict[str, float | bool]:
    """Axial-compression buckling of the interstage under full booster thrust.

    See SOURCES["iv1_interstage_buckling"]. Evaluated at the SMALLER radius, which carries the
    higher stress for a given thickness and is therefore critical.

    Returns the applied stress, the knocked-down allowable, the margin, and whether it passes.
    A negative margin means the interstage would collapse under its own thrust load, which no
    amount of mass-statement bookkeeping would reveal on its own.
    """
    mat = MATERIALS["airframe_al7075"]
    R = 0.5 * min(dv.booster.D, dv.payload_stage.D)
    t = dv.t_interstage
    F = dv.booster.F_thrust

    area = 2.0 * math.pi * R * t
    sigma_applied = F / area if area > 0.0 else float("inf")
    sigma_classical = 0.605 * mat.E * t / R if R > 0.0 else 0.0
    sigma_allow = BUCKLING_KNOCKDOWN * sigma_classical

    # yielding is the other failure mode; the lower allowable governs
    sigma_governing = min(sigma_allow, mat.sigma_yield)
    margin = sigma_governing / (BUCKLING_SF * sigma_applied) - 1.0 if sigma_applied > 0 else 0.0

    return {
        "sigma_applied": sigma_applied,
        "sigma_classical": sigma_classical,
        "sigma_allowable": sigma_allow,
        "sigma_governing": sigma_governing,
        "mode": "buckling" if sigma_allow < mat.sigma_yield else "yield",
        "margin": margin,
        "passes": margin >= 0.0,
        "t_required": BUCKLING_SF * F / (2.0 * math.pi * R * sigma_governing)
        if sigma_governing > 0.0
        else float("inf"),
    }


# --------------------------------------------------------------------------------------
#   Geometry
# --------------------------------------------------------------------------------------


def strake_mass(st: StrakeSpec) -> float:
    """Mass of all strake panels, kg."""
    v = st.n * st.area_one_side * st.thickness * STRAKE_AREA_TO_VOLUME
    return v * MATERIALS["fin_ti64"].density


def interstage_mass(dv: StackDesignVector) -> tuple[float, dict[str, float]]:
    """Truncated conical shell plus ring frames, kg. SOURCES["iv1_interstage_mass"]."""
    r1 = 0.5 * dv.booster.D
    r2 = 0.5 * dv.payload_stage.D
    L = dv.L_interstage
    mat = MATERIALS["airframe_al7075"]

    slant = math.hypot(L, r1 - r2)
    area = math.pi * (r1 + r2) * slant
    m_shell = area * dv.t_interstage * mat.density

    # ring frames at each end, annular discs of BULKHEAD_T_FRAC of the shell thickness
    m_rings = 0.0
    for r in (r1, r2):
        m_rings += math.pi * r * r * BULKHEAD_T_FRAC * dv.t_interstage * mat.density
    m_rings *= N_RING_INTERSTAGE / 2.0

    return m_shell + m_rings, {"shell": m_shell, "rings": m_rings, "slant_area": area}


def stage_geometry(dv: StackDesignVector, stage: StageSpec) -> dict[str, float]:
    """Closed-form areas and volumes for one stage, used until nTop measures it."""
    R = 0.5 * stage.D
    is_payload = stage is dv.payload_stage

    if is_payload:
        L_nose = dv.L_nose
        # A splined nose is measured from the SAME chord polygon nTop revolves, so the
        # analytic fallback and the measured solid describe one geometry. At the
        # ogive-equivalent control values the two agree to better than 1e-3 relative.
        nose_control = getattr(dv, "nose_control", None)
        if nose_control is not None:
            from ..oml_spline import SplineProfile

            _nose = SplineProfile(length=L_nose, radius=R, control=nose_control, n_poly=160)
            a_nose = _nose.lateral_area()
            v_nose = _nose.volume()
        else:
            a_nose = _tangent_ogive_surface_area(L_nose, R)
            v_nose = _tangent_ogive_volume(L_nose, R)
        L_cyl = stage.L - L_nose
    else:
        L_nose = 0.0
        a_nose = v_nose = 0.0
        L_cyl = stage.L

    a_cyl = 2.0 * math.pi * R * max(L_cyl, 0.0)
    v_cyl = math.pi * R * R * max(L_cyl, 0.0)
    a_fin = 2.0 * stage.n_fin * stage.S_fin_exposed
    a_strake = dv.strakes.wetted_area if is_payload else 0.0

    return {
        "area_wetted_nose": a_nose,
        "area_wetted_cyl": a_cyl,
        "area_wetted_body": a_nose + a_cyl,
        "area_wetted_fins": a_fin,
        "area_wetted_strakes": a_strake,
        "area_base": stage.S_ref,
        "volume_nose": v_nose,
        "volume_cyl": v_cyl,
        "volume_total": v_nose + v_cyl,
        "L_cyl": max(L_cyl, 0.0),
        "L_nose": L_nose,
    }


# --------------------------------------------------------------------------------------
#   Per-stage statement
# --------------------------------------------------------------------------------------


@dataclass
class StackMasses:
    """One `MassBuildup` per stage, plus the derived staging quantities."""

    per_stage: dict[int, MassBuildup] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    interstage: float = 0.0
    interstage_x: float = 0.0
    interstage_buckling: dict = field(default_factory=dict)

    # --- totals ---

    @property
    def m0(self) -> float:
        """Launch mass of the whole stack, kg."""
        return sum(mb.total for mb in self.per_stage.values()) + self.interstage

    def stage_total(self, index: int) -> float:
        return self.per_stage[index].total

    @property
    def x_cg(self) -> float:
        """Stack centre of gravity at launch, m from the stacked nose tip."""
        m = self.m0
        if m <= 0.0:
            return 0.0
        num = sum(mb.total * mb.x_cg for mb in self.per_stage.values())
        num += self.interstage * self.interstage_x
        return num / m

    def jettisoned_mass(self, booster_index: int = 1) -> float:
        """Mass that leaves at separation: the burnt-out booster plus the interstage.

        The booster's propellant is already gone by then, so only its inert mass counts.
        """
        mb = self.per_stage[booster_index]
        inert, _ = mb.excluding(*_propellant_names(mb))
        return inert + self.interstage

    def mass_after_separation(self, booster_index: int = 1) -> float:
        return self.m0 - self.jettisoned_mass(booster_index) - _burned(self.per_stage[booster_index])

    @property
    def measured_fraction(self) -> float:
        m = self.m0
        if m <= 0.0:
            return 0.0
        num = sum(
            e.mass
            for mb in self.per_stage.values()
            for e in mb.entries
            if e.provenance == "ntop_measured"
        )
        return num / m

    def table_rows(self) -> list[tuple[int, str, float, float, str]]:
        """Report-ready rows: (stage, item, mass_kg, station_m, provenance)."""
        rows: list[tuple[int, str, float, float, str]] = []
        for idx in sorted(self.per_stage):
            for e in sorted(self.per_stage[idx].entries, key=lambda q: -q.mass):
                rows.append((idx, e.name, e.mass, e.x_cg, e.provenance))
        if self.interstage > 0.0:
            rows.append((0, "Interstage", self.interstage, self.interstage_x, "analytic"))
        return rows


def _propellant_names(mb: MassBuildup) -> tuple[str, ...]:
    return tuple(e.name for e in mb.entries if e.name.startswith("Propellant"))


def _burned(mb: MassBuildup) -> float:
    return sum(e.mass for e in mb.entries if e.name.startswith("Propellant"))


# --------------------------------------------------------------------------------------
#   The build-up
# --------------------------------------------------------------------------------------


def build_stack_masses(
    dv: StackDesignVector,
    reqs: InterceptRequirements,
    meas: dict[int, NtopMeasurements] | None = None,
    motor: object | None = None,
) -> StackMasses:
    """Assemble one group-weight statement per stage.

    Stations are measured from each STAGE's own nose or forward face, then offset into stack
    coordinates for the stack centre of gravity. Keeping the per-stage statements in stage
    coordinates is what lets the stability analysis be done twice, once per flight configuration.
    """
    sm = StackMasses()
    meas = meas or {}

    # forward face of each stage in stack coordinates: payload stage first, then interstage,
    # then the booster
    x_cursor = 0.0
    offsets: dict[int, float] = {}
    for stage in reversed(dv.stages):          # payload stage is last in burn order, first in space
        offsets[stage.index] = x_cursor
        x_cursor += stage.L
        if stage is dv.payload_stage:
            sm.interstage_x = x_cursor + 0.5 * dv.L_interstage
            x_cursor += dv.L_interstage

    m_inter, inter_detail = interstage_mass(dv)
    sm.interstage = m_inter

    for stage in dv.stages:
        is_payload = stage is dv.payload_stage
        geo = stage_geometry(dv, stage)
        m = meas.get(stage.index)
        mb = MassBuildup()
        x0 = offsets[stage.index]

        # --- payload, only on the surviving stage: a requirement, not an estimate ---
        if is_payload:
            x_pay = geo["L_nose"] + dv.L_seeker + 0.5 * dv.L_payload_bay
            mb.add("Payload", reqs.m_payload, x_pay, "requirement", "SPEC_IV1 A5")
            a_nose = geo["area_wetted_nose"]
            m_radome = a_nose * T_RADOME * MATERIALS["radome_pyroceram"].density
            mb.add(
                "Radome",
                m_radome,
                0.45 * geo["L_nose"],
                "analytic",
                f"ogive area {a_nose:.4f} m^2 at {T_RADOME*1e3:.1f} mm; thickness is a guess",
            )

        # --- airframe: prefer nTop ---
        if m is not None and m.mass_structure is not None:
            x_struct = m.cg_structure[0] if m.cg_structure else 0.55 * stage.L
            mb.add(
                "Airframe structure and surfaces",
                m.mass_structure,
                x_struct,
                "ntop_measured",
                f"nTop structure volume {m.volume_structure} m^3",
            )
        else:
            a_shell = geo["area_wetted_cyl"]
            m_shell = a_shell * stage.t_wall * MATERIALS["airframe_al7075"].density
            m_fins = (
                stage.n_fin * stage.S_fin_exposed * stage.t_fin * 0.65
                * MATERIALS["fin_ti64"].density
            )
            m_bulk = (
                N_BULKHEAD * stage.S_ref * BULKHEAD_T_FRAC * stage.t_wall
                * MATERIALS["airframe_al7075"].density
            )
            mb.add("Airframe shell", m_shell, geo["L_nose"] + 0.5 * geo["L_cyl"], "analytic")
            mb.add("Fins", m_fins, stage.L - 0.55 * stage.c_r_fin, "analytic",
                   "0.65 area-to-volume factor for a tapered section")
            mb.add("Bulkheads and hardware", m_bulk, 0.55 * stage.L, "analytic", "guess")
            if is_payload:
                st = dv.strakes
                mb.add(
                    "Strakes",
                    strake_mass(st),
                    st.x_le + 0.5 * st.length,
                    "analytic",
                    f"{st.n} panels, {st.height*1e3:.0f} mm x {st.length:.2f} m, "
                    f"{STRAKE_AREA_TO_VOLUME} area-to-volume factor is a guess",
                )
            sm.warnings.append(
                f"stage {stage.index} airframe mass is analytic; no nTop measurement supplied"
            )

        # --- motor ---
        L_motor = _motor_bay_length(dv, stage, geo)
        if L_motor <= 0.0:
            raise ValueError(f"stage {stage.index}: no room for a motor ({L_motor:.3f} m)")
        x_motor = stage.L - 0.5 * L_motor

        if motor is not None and hasattr(motor, "inert_mass_breakdown"):
            bd = motor.inert_mass_breakdown(stage.index)      # type: ignore[attr-defined]
            m_case = bd.get("case", 0.0)
            m_ins = bd.get("insulation", 0.0)
            m_noz = bd.get("nozzle", 0.0)
            m_ign = bd.get("igniter", 0.0)
            corr_lo = bd.get("correlation_min")
            t_case = bd.get("case_thickness")
        else:
            m_case, case_detail = _case_for_stage(stage, L_motor)
            t_case = case_detail["t_case"]
            R_ins = 0.5 * stage.D - stage.t_wall
            m_ins = (2.0 * math.pi * R_ins * T_INSULATION * L_motor) * MATERIALS[
                "insulation_epdm"
            ].density
            m_noz = 0.12 / (1.0 - 0.12) * (m_case + m_ins)
            m_ign = 0.005 * stage.m_propellant
            corr_lo = stage.m_propellant * (1.0 / 0.92 - 1.0)

        mb.add("Motor case", m_case, x_motor, "analytic",
               f"t = {t_case*1e3:.2f} mm" if t_case else "")
        mb.add("Motor insulation", m_ins, x_motor, "analytic")
        mb.add("Nozzle assembly", m_noz, stage.L - 0.05 * stage.L, "correlation")
        mb.add("Igniter", m_ign, x_motor - 0.4 * L_motor, "analytic", "guess")

        # same correlation-floor policy as the single-body build-up
        inert = m_case + m_ins + m_noz + m_ign
        if corr_lo is not None and inert < corr_lo:
            mb.add(
                "Motor hardware not modelled",
                corr_lo - inert,
                x_motor,
                "correlation",
                "bottom-up sum {:.1f} kg raised to the tactical-motor correlation floor "
                "{:.1f} kg".format(inert, corr_lo),
            )
            sm.warnings.append(
                f"stage {stage.index}: motor inert raised from {inert:.1f} kg to the "
                f"correlation floor {corr_lo:.1f} kg"
            )

        mb.add("Propellant", stage.m_propellant, x_motor, "analytic")

        # --- dry contingency on this stage's dry mass ---
        dry_names = {e.name for e in mb.entries if e.name not in ("Payload", "Propellant")}
        m_dry = mb.subset(*dry_names)
        x_dry = (
            sum(e.mass * e.x_cg for e in mb.entries if e.name in dry_names) / m_dry
            if m_dry > 0.0
            else 0.5 * stage.L
        )
        mb.add(
            "Dry-mass contingency",
            DRY_MARGIN * m_dry,
            x_dry,
            "correlation",
            f"{DRY_MARGIN*100:.0f} percent on {m_dry:.1f} kg, AIAA S-120A Class-I",
        )

        # --- volume closure for this stage ---
        v_needed = stage.m_propellant / MATERIALS["propellant_htpb_ap"].density
        if is_payload:
            v_needed += reqs.m_payload / RHO_PAYLOAD
        v_avail = (m.volume_cavity if (m and m.volume_cavity) else geo["volume_total"] * 0.86)
        if m is None or m.volume_cavity is None:
            sm.warnings.append(
                f"stage {stage.index} cavity volume is an analytic 86 percent of enclosed volume"
            )
        if v_needed > v_avail / 1.05:
            sm.warnings.append(
                "stage {}: volume closure FAILS, need {:.4f} m^3, have {:.4f} m^3 with a "
                "5 percent packing margin".format(stage.index, v_needed, v_avail / 1.05)
            )

        # shift into stack coordinates
        shifted = MassBuildup(warnings=list(mb.warnings))
        for e in mb.entries:
            shifted.add(e.name, e.mass, x0 + e.x_cg, e.provenance, e.note)
        sm.per_stage[stage.index] = shifted

    sm.warnings.append(
        "interstage {:.2f} kg (shell {:.2f}, rings {:.2f}); separation hardware is not costed".format(
            m_inter, inter_detail["shell"], inter_detail["rings"]
        )
    )

    # The interstage carries the full booster thrust in compression. A mass statement alone would
    # never reveal that the member is too thin to hold the load it exists to transmit.
    buck = interstage_buckling_check(dv)
    sm.interstage_buckling = buck
    if not buck["passes"]:
        sm.warnings.append(
            "interstage BUCKLING FAILS in {} at {:.0f} kN booster thrust: applied {:.0f} MPa "
            "against a governing allowable of {:.0f} MPa, margin {:+.2f}. Needs at least "
            "{:.2f} mm of wall, against {:.2f} mm specified.".format(
                buck["mode"],
                dv.booster.F_thrust / 1e3,
                buck["sigma_applied"] / 1e6,
                buck["sigma_governing"] / 1e6,
                buck["margin"],
                buck["t_required"] * 1e3,
                dv.t_interstage * 1e3,
            )
        )
    return sm


def _motor_bay_length(dv: StackDesignVector, stage: StageSpec, geo: dict[str, float]) -> float:
    if stage is dv.payload_stage:
        return stage.L - geo["L_nose"] - dv.L_seeker - dv.L_payload_bay
    # booster: allow a short aft closure for the nozzle throat and attach ring
    return stage.L - 0.08 * stage.L


def _case_for_stage(stage: StageSpec, L_motor: float) -> tuple[float, dict[str, float]]:
    """Reuse the validated hoop-stress case sizing from masses.py.

    `motor_case_mass` takes the single-body `DesignVector`, so a light shim carries the three
    fields it actually reads. This keeps one implementation of the minimum-gauge floor rather than
    two that can drift apart.
    """

    class _Shim:
        D = stage.D
        t_wall = stage.t_wall
        p_c = stage.p_c

    return motor_case_mass(_Shim(), L_motor, "motorcase_cfrp")   # type: ignore[arg-type]


def static_margin_stage(x_cp: float, x_cg: float, D: float) -> float:
    """Static margin in calibres for one flight configuration. Positive is stable."""
    return (x_cp - x_cg) / D
