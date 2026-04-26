import numpy as np
from typing import Tuple

from config import (
    CHINCHILLA_E, CHINCHILLA_A, CHINCHILLA_B,
    CHINCHILLA_ALPHA, CHINCHILLA_BETA,
    CHANCE_LOSS, PARAM_GROWTH_WEEKLY,
    S0, GINI_COEFF,
    BYTES_PER_TOKEN, BYTES_PER_PB,
    LAMBDA0, BIORISK_THETA, ALPHA_MAX, BETA_MAX, DEATHS_BASE,
    REG_STEP,
)

# ─────────────────────────────────────────────────────────────────────────────
# EQUATIONS (from the paper)
# ─────────────────────────────────────────────────────────────────────────────


def chinchilla_loss(N_billions: float, D_pb: float) -> float:
    """
    Equation (2): L(N,D) = E + A/N^alpha + B/D^beta
    (Hoffmann et al., 2022 — Chinchilla scaling law)

    Units expected by the original Chinchilla paper:
        N : number of parameters (not billions)
        D : number of tokens (not petabytes)

    We convert internally:
        N_billions * 1e9       → raw parameter count
        D_pb * 1e15 / 2        → token count (~2 bytes per token)

    max(..., 1e-9) protects against division by zero (epsilon clamping).
    """
    N = max(N_billions * 1e9, 1e-9)
    D = max(D_pb * BYTES_PER_PB / BYTES_PER_TOKEN, 1e-9)

    return (CHINCHILLA_E
            + CHINCHILLA_A / (N ** CHINCHILLA_ALPHA)
            + CHINCHILLA_B / (D ** CHINCHILLA_BETA))


def domain_knowledge_score(N_billions: float, D_pb: float) -> float:
    """
    a1 = 1 - L(N,D) / ln(100,000)

    Normalises Chinchilla loss against chance-level loss for a
    100,000-option forced choice. Clipped to [0, 1].
    """
    L = chinchilla_loss(N_billions, D_pb)
    return float(np.clip(1.0 - L / CHANCE_LOSS, 0.0, 1.0))


def planning_ability_score(cumulative_dev_weeks: int) -> float:
    """
    Gompertz curve:
        S(Σdev) = exp(-A_gom * exp(-k * Σdev))
        A_gom = ln(1/S0)
        k     = ln(2) / (30 * A_gom)   [doubles every ~7 months ≈ 30 weeks]

    S0 is recalibrated to 0.01 for the 260-week simulation window.
    The original paper value (S0 ≈ 4.25e-8) keeps a2 ≈ 0 throughout
    the simulation, making planning ability effectively invisible.
    """
    A_gom = np.log(1.0 / S0)
    k_gom = np.log(2) / (30 * A_gom)
    t = max(cumulative_dev_weeks, 0)
    return float(np.exp(-A_gom * np.exp(-k_gom * t)))


def ai_potency(p0: float,
               a1: float, a2: float,
               w1: float, w2: float,
               omega1: float, omega2: float,
               reg_dom: float, reg_int: float) -> float:
    """
    Equation (1):
        Pₜ = P₀ + (100(w₁a₁^ω₁ + w₂a₂^ω₂) − P₀)(1 − max{reg_dom, reg_int})

    The max{} takes the stricter of domestic and international regulation.
    Regulations do not stack — only the binding constraint applies.
    """
    raw = 100.0 * (w1 * (a1 ** omega1) + w2 * (a2 ** omega2))
    reg_factor = 1.0 - max(reg_dom, reg_int)
    return float(p0 + (raw - p0) * reg_factor)


def data_availability(cumulative_dev_weeks: int,
                      D0: float, D_max: float) -> float:
    """
    Logistic curve:
        D(Σdev) = D_max / (1 + C * exp(-Σdev))
        C = D0 / (D_max - D0)

    Captures rapid early data growth that saturates as the finite
    global data pool is exhausted.
    """
    C = D0 / max(D_max - D0, 1e-9)
    t = max(cumulative_dev_weeks, 0)
    return float(D_max / (1.0 + C * np.exp(-t)))


def gini_update(G0: float, p_t: float, p0: float,
                p_max_global: float, reg_dom: float) -> float:
    """
    Equation (4):
        Gₜ = G₀ + 1.47 · (log(pₜ) − log(p₀)) / MAX(log(pⁿ)) · (1 − reg_dom)

    Calibrated from Bordot & Lorentz (2021) labour automation simulation data.
    Domestic regulation dampens the inequality effect.
    Clipped to [0, 1].
    """
    if p_max_global <= 0 or p0 <= 0 or p_t <= 0:
        return G0
    log_growth = (np.log(p_t) - np.log(p0)) / max(np.log(p_max_global), 1e-9)
    delta = GINI_COEFF * log_growth * (1.0 - reg_dom)
    return float(np.clip(G0 + delta, 0.0, 1.0))


def welfare(gini: float, gdp_per_capita: float) -> float:
    """
    Corrected welfare proxy (suggested by Pepijn Cobben):
        welfare = (1 − Gini) × GDP_per_capita

    Original CARMA used only Gini (w_t = Gini_t), which ignores
    absolute living standards. This proxy captures both distributional
    equity and aggregate wealth simultaneously.
    """
    return (1.0 - gini) * gdp_per_capita


def biorisk_factors(p_max: float) -> Tuple[float, float]:
    """
    Factor A — democratisation of access (sigmoid):
        A = 1 / (1 + exp(-(p_max - θ) / 100))
        AI lowers barriers for low-skill actors.

    Factor B — ceiling elevation (linear above threshold):
        B = max(0, (p_max - θ) / 100) if p_max >= θ else 0
        AI raises the maximum achievable severity for skilled actors.

    Both depend on max_n(p_t^n) — the global AI frontier, not the average.
    θ = 50 is the potency threshold above which dangerous capabilities emerge.
    """
    A = 1.0 / (1.0 + np.exp(-(p_max - BIORISK_THETA) / 100.0))
    B = max(0.0, (p_max - BIORISK_THETA) / 100.0) if p_max >= BIORISK_THETA else 0.0
    return float(A), float(B)


def biorisk_attacks_per_week(p_max: float, reg_int: float) -> float:
    """
    Equation (15): λ = λ₀(1 + αA + βB)(1 − reg_int)

    λ₀ = 0.76/52 attacks/week — base rate from Global Terrorism Database
    (38 bioterrorist incidents over 50 years).
    International regulation is the only lever that reduces global biorisk.
    """
    A, B = biorisk_factors(p_max)
    return float(LAMBDA0 * (1.0 + ALPHA_MAX * A + BETA_MAX * B) * (1.0 - reg_int))


def biorisk_severity(p_max: float) -> float:
    """
    Expected deaths per attack: λ₁(1 + βB)

    λ₁ = 5/38 — calibrated from the 2001 anthrax letters,
    the only post-1970 fatal bioterrorist attack (5 deaths, 38 total incidents).
    """
    _, B = biorisk_factors(p_max)
    return float(DEATHS_BASE * (1.0 + BETA_MAX * B))