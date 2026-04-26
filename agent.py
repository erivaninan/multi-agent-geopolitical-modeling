from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED AGENT (placeholder for LLM agents)
# ─────────────────────────────────────────────────────────────────────────────
#
# CARMA uses LLM agents with country-specific prompts written by international
# relations experts. Behavior emerges from the LLM's latent geopolitical
# knowledge — there are no fixed rules or hard-coded payoffs.
#
# This module replaces that mechanism with simple rule-based strategies
# until the actual LLM agents are integrated.
#
# All numerical thresholds below are arbitrary and chosen to produce
# qualitatively distinct behaviors between the three strategies.
# They are not calibrated on empirical data.
#
# Strategy descriptions:
#
#   hawkish  — inspired by a state that prioritizes strategic dominance.
#              Always develops. Resists regulation — starts deregulating
#              if domestic regulation begins to constrain AI growth (> 0.3)
#              or if international regulation rises above a minimal level (> 0.2).
#
#   dovish   — inspired by EU-like behavior: develops with safeguards.
#              Stops developing above p = 60 (arbitrary ceiling).
#              Regulates domestically up to 0.5 (arbitrary target).
#              Always pushes for international regulation.
#
#   balanced — reactive: develops only if lagging more than 15% behind the
#              strongest rival (threshold: 0.85 × p_max_rival).
#              Regulates moderately up to 0.3 domestically.
#              Pushes for international regulation if it is still low (< 0.4).
# ─────────────────────────────────────────────────────────────────────────────


def rule_based_action(strategy: str,
                      p: float, reg_dom: float, reg_int: float,
                      p_max_rival: float, t: int) -> Tuple[str, str, str]:
    """
    Returns (dev_action, reg_dom_action, reg_int_action).

    dev_action      : "develop" | "wait"
    reg_dom_action  : "regulate_dom" | "wait" | "deregulate_dom"
    reg_int_action  : "regulate_int" | "wait" | "deregulate_int"
    """
    if strategy == "hawkish":
        dev   = "develop"
        r_dom = "deregulate_dom" if reg_dom > 0.3 else "wait"
        r_int = "deregulate_int" if reg_int > 0.2 else "wait"

    elif strategy == "dovish":
        dev   = "develop" if p < 60 else "wait"
        r_dom = "regulate_dom" if reg_dom < 0.5 else "wait"
        r_int = "regulate_int"

    else:  # balanced
        behind = p < p_max_rival * 0.85
        dev    = "develop" if behind else "wait"
        r_dom  = "regulate_dom" if reg_dom < 0.3 else "wait"
        r_int  = "regulate_int" if reg_int < 0.4 else "wait"

    return dev, r_dom, r_int