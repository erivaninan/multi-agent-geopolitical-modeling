import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from config import CountryConfig
from equations import (
    domain_knowledge_score,
    planning_ability_score,
    ai_potency,
    data_availability,
    gini_update,
    welfare,
    PARAM_GROWTH_WEEKLY,
    REG_STEP,
)


# ─────────────────────────────────────────────────────────────────────────────
# COUNTRY STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryState:
    """
    Represents the full state of one country at any point in the simulation.

    State vector Sᵢ = (p, reg_dom, gini) as defined in the paper.
    GDP per capita is added to support the corrected welfare proxy.

    The step() method advances the state by one week given:
        - the country's chosen actions (dev_action, reg_dom_action)
        - the current global state (reg_int, p_max_global)
    """

    config: CountryConfig

    # Initial global regulation index — passed from simulation at init
    # so that the equation-derived p at t=0 is consistent with step 1.
    reg_int_init: float = 0.1

    # ── Dynamic state ─────────────────────────────────────────────────────

    # Sᵢ = (p, reg_dom, gini)
    p: float = 0.0        # AI potency (1–100)
    reg_dom: float = 0.0  # domestic regulation index [0, 1]
    gini: float = 0.0     # Gini inequality coefficient [0, 1]

    # Equation-derived p at t=0, used as the reference for Gini updates.
    # cfg.p0 is a calibration anchor (e.g. 75 for USA), not the true
    # starting p produced by the equations — using cfg.p0 as reference
    # would create an immediate Gini spike at week 1.
    p_initial: float = 0.0

    # Extended state — added to support corrected welfare proxy
    gdp_per_capita: float = 0.0  # GDP per capita in USD

    # Intermediate variables — not formally part of Sᵢ in the paper,
    # but necessary to compute AI potency at each step.
    N_billions: float = 0.0        # model parameter count (billions)
    D_pb: float = 0.0              # training data available (petabytes)
    cumulative_dev_weeks: int = 0  # Σ_t {dev}_t^n

    # ── History (one entry per week, appended in record()) ────────────────

    history_p: List[float] = field(default_factory=list)
    history_reg_dom: List[float] = field(default_factory=list)
    history_gini: List[float] = field(default_factory=list)
    history_welfare: List[float] = field(default_factory=list)
    history_gdp: List[float] = field(default_factory=list)
    history_a1: List[float] = field(default_factory=list)
    history_a2: List[float] = field(default_factory=list)
    history_N: List[float] = field(default_factory=list)
    history_D: List[float] = field(default_factory=list)
    history_actions: List[Tuple[str, str]] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        """
        Initialise dynamic state from CountryConfig.
        Called automatically by the dataclass after __init__.

        p is computed from the equations rather than taken directly from
        cfg.p0, so that the value at t=0 is consistent with step 1 and
        no artificial jump appears in the plots.

        record() is NOT called here — week 0 is recorded explicitly
        in simulation.run() before the main loop starts.
        """
        cfg = self.config

        self.reg_dom        = cfg.reg_dom_0
        self.gini           = cfg.gini_0
        self.gdp_per_capita = cfg.gdp_per_capita_0
        self.N_billions     = cfg.N0_billions

        # Initial data stock D₀ⁿ (paper equation after eq. 3):
        # D₀ⁿ = D₀ᵈ + w₁ⁿ(D₀ᵍ − w₂ⁿ D₀ᵈ)
        D0_d      = cfg.D0_domestic_pb
        D0_g      = cfg.D_global_pb
        self.D_pb = D0_d + cfg.w1_data * (D0_g - cfg.w2_data * D0_d)
        self.D_pb = float(np.clip(self.D_pb, 0.0, cfg.D_max_pb))

        # Compute p from the equations at t=0
        a1     = domain_knowledge_score(self.N_billions, self.D_pb)
        a2     = planning_ability_score(0)  # 0 development weeks at start
        self.p = ai_potency(
            cfg.p0, a1, a2,
            cfg.w1, cfg.w2,
            cfg.omega1, cfg.omega2,
            self.reg_dom, self.reg_int_init
        )

        # Store equation-derived p as the Gini reference baseline.
        # This prevents a Gini spike at week 1 caused by the mismatch
        # between cfg.p0 (e.g. 75) and the equation-derived p (e.g. 36).
        self.p_initial = self.p

    # ─────────────────────────────────────────────────────────────────────

    def record(self):
        """
        Append current state to history lists.
        Called at the end of each step() — not at initialisation.
        Week 0 is recorded explicitly in simulation.run().
        """
        self.history_p.append(self.p)
        self.history_reg_dom.append(self.reg_dom)
        self.history_gini.append(self.gini)
        self.history_welfare.append(welfare(self.gini, self.gdp_per_capita))
        self.history_gdp.append(self.gdp_per_capita)
        self.history_N.append(self.N_billions)
        self.history_D.append(self.D_pb)

    # ─────────────────────────────────────────────────────────────────────

    def step(self,
             dev_action: str,
             reg_dom_action: str,
             reg_int: float,
             p_max_global: float):
        """
        Advance the country state by one week.

        Parameters
        ----------
        dev_action      : "develop" | "wait"
        reg_dom_action  : "regulate_dom" | "wait" | "deregulate_dom"
        reg_int         : current global international regulation index [0,1]
        p_max_global    : maximum AI potency among all countries this week
                          (needed for Gini equation normalisation)
        """
        cfg = self.config

        # ── Step 1: Update cumulative development counter ─────────────────
        if dev_action == "develop":
            self.cumulative_dev_weeks += 1

        # ── Step 2: Update parameter count N ─────────────────────────────
        # Equation (3): Nⁿ(t) = N₀ · (⁵²√3.7)^Σdev
        # Implemented incrementally: N is multiplied by the weekly growth
        # factor each time the country chooses "develop".
        if dev_action == "develop":
            self.N_billions *= PARAM_GROWTH_WEEKLY

        # ── Step 3: Update data availability D (logistic curve) ───────────
        # D grows toward D_max as the country accumulates development weeks.
        self.D_pb = data_availability(
            self.cumulative_dev_weeks,
            self.D_pb,
            cfg.D_max_pb
        )

        # ── Step 4: Compute a1 and a2 ────────────────────────────────────
        # a1: domain knowledge score via Chinchilla scaling law
        a1 = domain_knowledge_score(self.N_billions, self.D_pb)

        # a2: planning ability score via Gompertz curve
        a2 = planning_ability_score(self.cumulative_dev_weeks)

        self.history_a1.append(a1)
        self.history_a2.append(a2)

        # ── Step 5: Update domestic regulation ───────────────────────────
        if reg_dom_action == "regulate_dom":
            self.reg_dom = float(np.clip(self.reg_dom + REG_STEP, 0.0, 1.0))
        elif reg_dom_action == "deregulate_dom":
            self.reg_dom = float(np.clip(self.reg_dom - REG_STEP, 0.0, 1.0))
        # "wait" → reg_dom unchanged

        # ── Step 6: Update AI potency ─────────────────────────────────────
        # Equation (1):
        # Pₜ = P₀ + (100(w₁a₁^ω₁ + w₂a₂^ω₂) − P₀)(1 − max{reg_dom, reg_int})
        self.p = ai_potency(
            p0=cfg.p0,
            a1=a1, a2=a2,
            w1=cfg.w1, w2=cfg.w2,
            omega1=cfg.omega1, omega2=cfg.omega2,
            reg_dom=self.reg_dom,
            reg_int=reg_int
        )

        # ── Step 7: Update Gini inequality ───────────────────────────────
        # Equation (4):
        # Gₜ = G₀ + 1.47 · (log(pₜ)−log(p_initial)) / MAX(log(p)) · (1−reg_dom)
        # p_initial (equation-derived) is used instead of cfg.p0 to avoid
        # an artificial Gini spike caused by their mismatch.
        self.gini = gini_update(
            G0=cfg.gini_0,
            p_t=self.p,
            p0=self.p_initial,
            p_max_global=p_max_global,
            reg_dom=self.reg_dom
        )

        # ── Step 8: Update GDP per capita ─────────────────────────────────
        # Not in the original CARMA paper — added to support the corrected
        # welfare proxy: welfare = (1 − Gini) × GDP per capita.
        self.gdp_per_capita *= (1.0 + cfg.gdp_growth_weekly)

        # ── Record actions and updated state ─────────────────────────────
        self.history_actions.append((dev_action, reg_dom_action))
        self.record()