"""Group-weight build-up for the SV-1 rocket.

Two paths feed this module:

- **nTop-measured**: when an `NtopMeasurements` is available, the airframe, fin and bulkhead
  masses come from real solid volumes multiplied by material density. This is the preferred
  path and is what closes the SPEC.md section 6 loop.
- **Analytic fallback**: closed-form shell volumes from `DesignVector`, used on the first
  iteration before any geometry exists, and whenever nTop did not report a quantity.

Every mass entry records which path produced it, so the report can show how much of the mass
statement is measured rather than estimated.

No mass may come from a bare number. Correlations are cited; guesses say "guess".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import (
    MATERIALS,
    DesignVector,
    MassStatement,
    Material,
    NtopMeasurements,
    Requirements,
    register_sources,
)

SOURCES: dict[str, str] = {
    "mass_frac_tactical_motor": (
        "Fleeman, Tactical Missile Design, 2nd ed., AIAA Education Series, Chapter 4: "
        "tactical solid motor propellant mass fraction typically 0.80 to 0.92 of motor "
        "assembly mass"
    ),
    "hoop_stress_case": (
        "Sutton and Biblarz, Rocket Propulsion Elements, 9th ed., Chapter 14: thin-wall "
        "pressure-vessel hoop stress t = p*R/sigma, with a design safety factor on burst"
    ),
    "case_safety_factor": (
        "MIL-STD-1522A / common solid-motor practice: burst safety factor 1.5 on MEOP for "
        "metallic cases, 2.0 for filament-wound composite"
    ),
    "case_minimum_gauge": (
        "Manufacturing minimum gauge, not a strength requirement. Hoop stress alone gives a "
        "sub-millimetre wall at this chamber pressure, which cannot be filament wound or "
        "machined. Floors of 1.8 mm (composite, roughly 3 to 4 wound plies) and 1.2 mm "
        "(metallic) are taken as engineering minimums. GUESS: no public minimum-gauge "
        "standard was found, but without the floor the model returns a motor mass fraction "
        "of 0.95, well above the 0.80 to 0.92 band for real tactical motors"
    ),
    "insulation_thickness": (
        "Sutton and Biblarz, Rocket Propulsion Elements, 9th ed., Chapter 14: EPDM chamber "
        "insulation 1 to 5 mm for short-burn tactical motors; 2.5 mm taken here as a "
        "mid-range modelling choice"
    ),
    "nozzle_mass_fraction": (
        "Fleeman, Tactical Missile Design, 2nd ed., Chapter 4: fixed nozzle assembly is "
        "typically 8 to 15 percent of inert motor mass for a tactical solid motor"
    ),
    "radome_thickness": (
        "GUESS: 6 mm Pyroceram radome wall. No public thickness standard was found for a "
        "0.35 m class radome; scaled from the requirement that wall thickness be an integer "
        "number of half-wavelengths at Ku band, which is not modelled here"
    ),
    "bulkhead_allowance": (
        "GUESS: four ring bulkheads, each a full-diameter disc of 40 percent of wall "
        "thickness. Not from a source; a placeholder for joint, rail and lug hardware"
    ),
    "avionics_density": (
        "Fleeman, Tactical Missile Design, 2nd ed., Chapter 2: missile electronics and "
        "actuation packaging density of order 800 to 1600 kg/m^3"
    ),
    "contingency_margin": (
        "AIAA S-120A-2015 mass-properties practice: conceptual-phase (Class-I) dry-mass "
        "growth allowance of 15 to 25 percent; 18 percent taken here"
    ),
}
register_sources(SOURCES)


# --------------------------------------------------------------------------------------
#   Modelling constants, all traceable to SOURCES above
# --------------------------------------------------------------------------------------

T_INSULATION = 0.0025           # m, EPDM chamber insulation. SOURCES["insulation_thickness"]
CASE_SF_COMPOSITE = 2.0         # SOURCES["case_safety_factor"]
CASE_SF_METALLIC = 1.5          # SOURCES["case_safety_factor"]
T_CASE_MIN_COMPOSITE = 0.0018   # m. SOURCES["case_minimum_gauge"]
T_CASE_MIN_METALLIC = 0.0012    # m. SOURCES["case_minimum_gauge"]
NOZZLE_FRAC_OF_INERT = 0.12     # SOURCES["nozzle_mass_fraction"], mid-range of 8 to 15 percent
T_RADOME = 0.006                # m. SOURCES["radome_thickness"] - a guess
N_BULKHEAD = 4                  # SOURCES["bulkhead_allowance"] - a guess
BULKHEAD_T_FRAC = 0.40          # SOURCES["bulkhead_allowance"] - a guess
RHO_AVIONICS = 1200.0           # kg/m^3. SOURCES["avionics_density"], mid-range
DRY_MARGIN = 0.18               # SOURCES["contingency_margin"]
IGNITER_MASS = 0.8              # kg. GUESS: small pyrogen igniter for a 0.35 m motor
SOURCES["igniter_mass"] = (
    "GUESS: 0.8 kg pyrogen igniter assembly for a 0.35 m diameter tactical motor. "
    "No source found; small relative to total and treated as fixed."
)


# The propellant line items, in burn order. Anything that iterates over "the propellant" must use
# this tuple rather than listing names inline, so that adding a burn phase cannot silently leave a
# charge out of the launch mass or out of the burnout mass. Leaving the terminal charge out of this
# list was a real defect: the launch mass came out 28 kg light.
PROPELLANT_ITEMS: tuple[str, ...] = (
    "Propellant, boost",
    "Propellant, sustain",
    "Propellant, terminal",
)


@dataclass
class MassEntry:
    """One line of the group-weight statement, with its provenance."""

    name: str
    mass: float                 # kg
    x_cg: float                 # m from the nose tip
    provenance: str             # "ntop_measured" | "analytic" | "requirement" | "correlation"
    note: str = ""


@dataclass
class MassBuildup:
    """The full mass statement plus provenance and the derived stability quantities."""

    entries: list[MassEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, mass: float, x_cg: float, provenance: str, note: str = "") -> None:
        if mass < 0.0:
            raise ValueError(f"negative mass for {name!r}: {mass}")
        self.entries.append(MassEntry(name, mass, x_cg, provenance, note))

    # --- totals ---

    @property
    def total(self) -> float:
        return sum(e.mass for e in self.entries)

    @property
    def x_cg(self) -> float:
        m = self.total
        return sum(e.mass * e.x_cg for e in self.entries) / m if m > 0.0 else 0.0

    def subset(self, *names: str) -> float:
        return sum(e.mass for e in self.entries if e.name in names)

    def excluding(self, *names: str) -> tuple[float, float]:
        """Return (mass, x_cg) with the named entries removed. Use for burnout state."""
        keep = [e for e in self.entries if e.name not in names]
        m = sum(e.mass for e in keep)
        x = sum(e.mass * e.x_cg for e in keep) / m if m > 0.0 else 0.0
        return m, x

    @property
    def measured_fraction(self) -> float:
        """Fraction of total mass that came from an nTop measurement rather than an estimate."""
        m = self.total
        if m <= 0.0:
            return 0.0
        return sum(e.mass for e in self.entries if e.provenance == "ntop_measured") / m

    def to_statement(self) -> MassStatement:
        st = MassStatement()
        for e in self.entries:
            st.add(e.name, e.mass, e.x_cg)
        return st

    def table_rows(self) -> list[tuple[str, float, float, float, str]]:
        """Report-ready rows: (name, mass_kg, percent_of_total, x_cg_m, provenance)."""
        tot = self.total
        rows = [
            (e.name, e.mass, 100.0 * e.mass / tot if tot > 0 else 0.0, e.x_cg, e.provenance)
            for e in self.entries
        ]
        rows.sort(key=lambda r: -r[1])
        return rows


# --------------------------------------------------------------------------------------
#   Analytic geometry helpers, used when nTop has not measured yet
# --------------------------------------------------------------------------------------


def _tangent_ogive_surface_area(L: float, R: float, n: int = 200) -> float:
    """Wetted area of a tangent ogive of length L and base radius R, by revolution quadrature.

    A tangent ogive of length L and base radius R has generating-circle radius
    rho = (R^2 + L^2) / (2R), and profile y(x) = sqrt(rho^2 - (L - x)^2) - (rho - R).
    """
    if R <= 0.0 or L <= 0.0:
        return 0.0
    rho = (R * R + L * L) / (2.0 * R)
    area = 0.0
    prev_x, prev_y = 0.0, 0.0
    for i in range(1, n + 1):
        x = L * i / n
        inner = rho * rho - (L - x) ** 2
        y = math.sqrt(max(inner, 0.0)) - (rho - R)
        y = max(y, 0.0)
        # frustum lateral area between the two stations
        slant = math.hypot(x - prev_x, y - prev_y)
        area += math.pi * (y + prev_y) * slant
        prev_x, prev_y = x, y
    return area


def _tangent_ogive_volume(L: float, R: float, n: int = 200) -> float:
    """Enclosed volume of a tangent ogive, by the same quadrature."""
    if R <= 0.0 or L <= 0.0:
        return 0.0
    rho = (R * R + L * L) / (2.0 * R)
    vol = 0.0
    prev_x, prev_y = 0.0, 0.0
    for i in range(1, n + 1):
        x = L * i / n
        inner = rho * rho - (L - x) ** 2
        y = max(math.sqrt(max(inner, 0.0)) - (rho - R), 0.0)
        dx = x - prev_x
        vol += math.pi * dx * (prev_y * prev_y + prev_y * y + y * y) / 3.0
        prev_x, prev_y = x, y
    return vol


def analytic_geometry(dv: DesignVector) -> dict[str, float]:
    """Closed-form areas and volumes for the SV-1 outer mould line.

    Used on iteration 0 and as the fallback for anything nTop does not report.
    """
    R = 0.5 * dv.D
    r_base = 0.5 * dv.d_base

    a_nose = _tangent_ogive_surface_area(dv.L_nose, R)
    v_nose = _tangent_ogive_volume(dv.L_nose, R)

    a_cyl = 2.0 * math.pi * R * dv.L_body_cyl
    v_cyl = math.pi * R * R * dv.L_body_cyl

    # conical boattail from R to r_base
    slant = math.hypot(dv.L_boattail, R - r_base)
    a_boat = math.pi * (R + r_base) * slant
    v_boat = math.pi * dv.L_boattail * (R * R + R * r_base + r_base * r_base) / 3.0

    # fins: both sides of every exposed panel
    a_fin = 2.0 * dv.n_fin * dv.S_fin_exposed

    return {
        "area_wetted_nose": a_nose,
        "area_wetted_cyl": a_cyl,
        "area_wetted_boattail": a_boat,
        "area_wetted_body": a_nose + a_cyl + a_boat,
        "area_wetted_fins": a_fin,
        "area_base": dv.S_base,
        "volume_nose": v_nose,
        "volume_cyl": v_cyl,
        "volume_boattail": v_boat,
        "volume_total": v_nose + v_cyl + v_boat,
    }


# --------------------------------------------------------------------------------------
#   Motor inert mass
# --------------------------------------------------------------------------------------


def motor_case_mass(
    dv: DesignVector,
    L_motor: float,
    material_key: str = "motorcase_cfrp",
) -> tuple[float, dict[str, float]]:
    """Motor case mass from thin-wall hoop stress at chamber pressure.

    t = SF * p_c * R / sigma_yield, with SF from SOURCES["case_safety_factor"].
    Returns (mass_kg, detail_dict). The wall thickness is reported so the caller can check it
    against the airframe wall thickness in `DesignVector`.
    """
    mat: Material = MATERIALS[material_key]
    sf = CASE_SF_COMPOSITE if "cfrp" in material_key else CASE_SF_METALLIC
    R_in = 0.5 * dv.D - dv.t_wall - T_INSULATION
    if R_in <= 0.0:
        raise ValueError("no room for a motor case inside the airframe")

    t_stress = sf * dv.p_c * R_in / mat.sigma_yield
    t_min = T_CASE_MIN_COMPOSITE if "cfrp" in material_key else T_CASE_MIN_METALLIC
    t_case = max(t_stress, t_min)
    gauge_limited = t_min > t_stress

    # cylindrical shell plus two hemispherical-equivalent closures
    v_cyl = 2.0 * math.pi * R_in * t_case * L_motor
    v_ends = 2.0 * (2.0 * math.pi * R_in * R_in * t_case)
    mass = (v_cyl + v_ends) * mat.density
    return mass, {
        "t_case": t_case,
        "t_stress": t_stress,
        "t_min_gauge": t_min,
        "gauge_limited": gauge_limited,
        "R_in": R_in,
        "safety_factor": sf,
        "material": material_key,
    }


def motor_inerts(
    dv: DesignVector,
    L_motor: float,
    material_key: str = "motorcase_cfrp",
) -> dict[str, float]:
    """Case, insulation, nozzle and igniter masses, plus a correlation cross-check.

    The physics route sizes the case from hoop stress and the insulation from a constant
    thickness. The nozzle is a fraction of the resulting inert mass. The correlation route
    applies a tactical-motor propellant mass fraction. Both are returned so the caller can
    compare them; the physics route is the one used in the mass statement.
    """
    m_case, case_detail = motor_case_mass(dv, L_motor, material_key)

    R_ins = 0.5 * dv.D - dv.t_wall
    v_ins = 2.0 * math.pi * R_ins * T_INSULATION * L_motor
    m_ins = v_ins * MATERIALS["insulation_epdm"].density

    m_structural = m_case + m_ins
    # nozzle taken as a fraction of total inert, so solve m_noz = f*(m_structural + m_noz)
    m_noz = NOZZLE_FRAC_OF_INERT / (1.0 - NOZZLE_FRAC_OF_INERT) * m_structural

    m_inert_physics = m_case + m_ins + m_noz + IGNITER_MASS

    m_prop = dv.m_p_boost + dv.m_p_sustain
    # correlation cross-check: propellant mass fraction 0.80 to 0.92 of motor assembly
    m_inert_corr_low = m_prop * (1.0 / 0.92 - 1.0)
    m_inert_corr_high = m_prop * (1.0 / 0.80 - 1.0)

    return {
        "case": m_case,
        "insulation": m_ins,
        "nozzle": m_noz,
        "igniter": IGNITER_MASS,
        "inert_total": m_inert_physics,
        "t_case": case_detail["t_case"],
        "t_case_stress": case_detail["t_stress"],
        "case_gauge_limited": case_detail["gauge_limited"],
        "propellant": m_prop,
        "mass_fraction_physics": m_prop / (m_prop + m_inert_physics),
        "inert_correlation_low": m_inert_corr_low,
        "inert_correlation_high": m_inert_corr_high,
    }


# --------------------------------------------------------------------------------------
#   The build-up
# --------------------------------------------------------------------------------------


def build_masses(
    dv: DesignVector,
    reqs: Requirements,
    meas: NtopMeasurements | None = None,
    case_material: str = "motorcase_cfrp",
    motor: object | None = None,
) -> MassBuildup:
    """Assemble the group-weight statement.

    Station convention: x measured aft from the nose tip, in metres.

    `motor` is an optional `rocketgen.sizing.propulsion.SolidMotor`. When supplied, its
    `inert_mass_breakdown()` is the authority for the motor inert masses and the local
    `motor_inerts()` estimate is not used. This matters: the motor model closes the grain
    geometry and nozzle sizing properly, so its case length and insulation area are real
    rather than assumed, and having two independent estimates of the same mass in one project
    is a defect. `motor_inerts()` remains for the no-motor case (iteration 0 and the unit tests)
    and as an independent cross-check, reported in the warnings when the two disagree by more
    than 25 percent.
    """
    mb = MassBuildup()
    geo = analytic_geometry(dv)

    x_seeker = 0.5 * dv.L_seeker
    x_guidance = dv.L_seeker + 0.5 * dv.L_guidance
    x_warhead = dv.L_seeker + dv.L_guidance + 0.5 * dv.L_warhead
    x_motor_fwd = dv.L_seeker + dv.L_guidance + dv.L_warhead
    L_motor = dv.L_total - x_motor_fwd - dv.L_boattail
    if L_motor <= 0.0:
        raise ValueError(f"no room for the motor: L_motor = {L_motor:.3f} m")
    x_motor = x_motor_fwd + 0.5 * L_motor

    # --- payload and avionics: these are requirements, not estimates ---
    mb.add("Warhead", reqs.m_warhead, x_warhead, "requirement", "SPEC R4")
    mb.add("Guidance, seeker, actuation", reqs.m_guidance, x_guidance, "requirement", "SPEC R5")

    # --- radome ---
    a_nose = geo["area_wetted_nose"]
    m_radome = a_nose * T_RADOME * MATERIALS["radome_pyroceram"].density
    mb.add(
        "Radome",
        m_radome,
        0.45 * dv.L_nose,
        "analytic",
        f"ogive area {a_nose:.4f} m^2 x {T_RADOME * 1e3:.1f} mm, thickness is a guess",
    )

    # --- airframe structure: prefer nTop ---
    if meas is not None and meas.mass_structure is not None:
        x_struct = meas.cg_structure[0] if meas.cg_structure else 0.55 * dv.L_total
        mb.add(
            "Airframe structure and fins",
            meas.mass_structure,
            x_struct,
            "ntop_measured",
            f"nTop solid volume {meas.volume_structure} m^3",
        )
    else:
        # analytic shell: body wall aft of the radome, plus fins, plus bulkheads
        a_shell = geo["area_wetted_cyl"] + geo["area_wetted_boattail"]
        m_shell = a_shell * dv.t_wall * MATERIALS["airframe_al7075"].density
        m_fins = dv.n_fin * dv.S_fin_exposed * dv.t_fin * 0.65 * MATERIALS["fin_ti64"].density
        m_bulk = (
            N_BULKHEAD
            * dv.S_ref
            * BULKHEAD_T_FRAC
            * dv.t_wall
            * MATERIALS["airframe_al7075"].density
        )
        mb.add("Airframe shell", m_shell, dv.L_nose + 0.5 * dv.L_body_cyl, "analytic")
        mb.add(
            "Fins",
            m_fins,
            dv.x_fin_le + 0.45 * dv.c_r_fin,
            "analytic",
            "0.65 area-to-volume factor for a tapered biconvex section",
        )
        mb.add("Bulkheads and hardware", m_bulk, 0.55 * dv.L_total, "analytic", "guess")
        mb.warnings.append("airframe mass is analytic; no nTop measurement supplied")

    # --- motor ---
    inerts = motor_inerts(dv, L_motor, case_material)

    if motor is not None:
        # The motor model is the authority. Cross-check the local estimate against it and warn
        # on disagreement rather than silently preferring one.
        bd = motor.inert_mass_breakdown()          # type: ignore[attr-defined]
        local, model = inerts["inert_total"], bd["total_physics"]
        if abs(local - model) / max(model, 1e-9) > 0.25:
            mb.warnings.append(
                "the two bottom-up motor inert estimates disagree: masses.py {:.1f} kg vs "
                "propulsion.py {:.1f} kg. The motor model governs.".format(local, model)
            )
        inerts = {
            "case": bd["case"],
            "insulation": bd["insulation"],
            "nozzle": bd["nozzle"],
            "igniter": bd["igniter"],
            "inert_total": bd["total_physics"],
            "t_case": bd["case_thickness"],
            "propellant": dv.m_p_boost + dv.m_p_sustain,
            "inert_correlation_low": bd["correlation_min"],
            "inert_correlation_high": bd["correlation_max"],
        }

    mb.add("Motor case", inerts["case"], x_motor, "analytic", f"t = {inerts['t_case']*1e3:.2f} mm")
    mb.add("Motor insulation", inerts["insulation"], x_motor, "analytic")
    mb.add("Nozzle assembly", inerts["nozzle"], dv.L_total - 0.4 * dv.L_boattail, "correlation")
    mb.add("Igniter", inerts["igniter"], x_motor_fwd + 0.05, "analytic", "guess")

    # The bottom-up case-plus-insulation-plus-nozzle sum is known to be incomplete at this
    # fidelity: it omits thrust skirts, case joints and closure hardware, the aft attachment
    # ring, the nozzle ablative liner and exit cone, and the blast tube. Left uncorrected it
    # returns a motor propellant mass fraction above the 0.80 to 0.92 band that real tactical
    # motors achieve, which would make the whole rocket optimistically light.
    #
    # Policy: the correlation band governs. When the bottom-up sum falls below the band, the
    # shortfall is booked as a single explicit line item and the shortfall is reported. This is
    # standard Class-I practice: use the bottom-up estimate where it is complete, and a
    # calibrated correlation where it is not.
    shortfall = inerts["inert_correlation_low"] - inerts["inert_total"]
    if shortfall > 0.0:
        mb.add(
            "Motor hardware not modelled",
            shortfall,
            x_motor,
            "correlation",
            "bottom-up sum {:.1f} kg raised to the tactical-motor correlation floor "
            "{:.1f} kg (mass fraction 0.92); covers skirts, joints, closures, nozzle "
            "liner and exit cone".format(inerts["inert_total"], inerts["inert_correlation_low"]),
        )
        mb.warnings.append(
            "motor inert mass raised from a bottom-up {:.1f} kg to the correlation floor "
            "{:.1f} kg; the bottom-up model is incomplete by {:.1f} kg".format(
                inerts["inert_total"], inerts["inert_correlation_low"], shortfall
            )
        )
    elif inerts["inert_total"] > inerts["inert_correlation_high"]:
        mb.warnings.append(
            "motor inert mass {:.1f} kg exceeds the correlation band upper bound {:.1f} kg; "
            "the case is strength-driven, not gauge-driven".format(
                inerts["inert_total"], inerts["inert_correlation_high"]
            )
        )

    # --- propellant ---
    # Laid out forward to aft in burn order (boost, sustain, terminal) so that the CG shift
    # through the burn is represented. The terminal pulse sits aft of the sustain charge, at the
    # nozzle end, which is why burning it moves the CG forward too.
    m_p = dv.m_p_boost + dv.m_p_sustain + dv.m_p_terminal
    if m_p > 0.0:
        x0 = x_motor_fwd
        for name, mass in (
            ("Propellant, boost", dv.m_p_boost),
            ("Propellant, sustain", dv.m_p_sustain),
            ("Propellant, terminal", dv.m_p_terminal),
        ):
            if mass <= 0.0:
                continue
            span = L_motor * mass / m_p
            mb.add(name, mass, x0 + 0.5 * span, "analytic")
            x0 += span

    # --- dry-mass contingency, applied to everything that is not payload or propellant ---
    dry_names = {
        e.name
        for e in mb.entries
        if e.name not in PROPELLANT_ITEMS and e.name != "Warhead"
    }
    m_dry = mb.subset(*dry_names)
    x_dry = (
        sum(e.mass * e.x_cg for e in mb.entries if e.name in dry_names) / m_dry
        if m_dry > 0.0
        else 0.5 * dv.L_total
    )
    mb.add(
        "Dry-mass contingency",
        DRY_MARGIN * m_dry,
        x_dry,
        "correlation",
        f"{DRY_MARGIN*100:.0f} percent on {m_dry:.1f} kg of dry mass, AIAA S-120A Class-I",
    )

    # --- volume closure check against the nTop cavity ---
    v_needed = (
        m_p / MATERIALS["propellant_htpb_ap"].density
        + reqs.m_warhead / 1750.0            # warhead packaging density, see note below
        + reqs.m_guidance / RHO_AVIONICS
    )
    SOURCES.setdefault(
        "warhead_density",
        "Fleeman, Tactical Missile Design, 2nd ed., Chapter 2: blast-fragmentation warhead "
        "packaging density of order 1700 to 1900 kg/m^3",
    )
    v_available = meas.volume_cavity if (meas and meas.volume_cavity) else None
    if v_available is None:
        # analytic cavity: outer volume less the shell, less the radome bay walls
        v_available = geo["volume_total"] * 0.86
        mb.warnings.append(
            "internal cavity volume is an analytic 86 percent of enclosed volume; "
            "no nTop measurement supplied"
        )
    if v_needed > v_available / 1.05:
        mb.warnings.append(
            "volume closure FAILS: need {:.4f} m^3, have {:.4f} m^3 with 5 percent packing "
            "margin".format(v_needed, v_available / 1.05)
        )

    return mb


def static_margin(x_cp: float, x_cg: float, D: float) -> float:
    """Static margin in calibres. Positive means statically stable (CP aft of CG)."""
    return (x_cp - x_cg) / D
