from typing import Tuple
import os
from openai import OpenAI

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


# ─────────────────────────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────────────────────────

def llm_action(country_name: str,
               p: float, reg_dom: float, reg_int: float,
               gini: float, gdp: float,
               p_max_rival: float, t: int,
               country_prompt: str) -> Tuple[str, str, str]:
    """
    Uses an OpenAI LLM agent to decide actions for a country.
    The API key is read from the OPENAI_API_KEY environment variable.
    Returns (dev_action, reg_dom_action, reg_int_action).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = f"""You are acting as {country_name} in a geopolitical AI arms race simulation.
{country_prompt}

You must respond with exactly three actions separated by commas, one from each category:
1. AI development: develop | wait
2. Domestic regulation: regulate_dom | wait | deregulate_dom
3. International regulation: regulate_int | wait | deregulate_int

Respond ONLY with the three actions, nothing else.
Example: develop, regulate_dom, wait"""

    user_prompt = f"""Week {t}. Current state of {country_name}:
- AI potency: {p:.2f} / 100
- Domestic regulation: {reg_dom:.2f}
- International regulation (global): {reg_int:.2f}
- Gini inequality: {gini:.3f}
- GDP per capita: ${gdp:,.0f}
- Max rival potency: {p_max_rival:.2f}

What are your three actions this week?"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=1.0,
        max_tokens=20
    )

    raw   = response.choices[0].message.content.strip().lower()
    parts = [p.strip() for p in raw.split(",")]

    valid_dev = {"develop", "wait"}
    valid_dom = {"regulate_dom", "wait", "deregulate_dom"}
    valid_int = {"regulate_int", "wait", "deregulate_int"}

    dev   = parts[0] if len(parts) > 0 and parts[0] in valid_dev else "wait"
    r_dom = parts[1] if len(parts) > 1 and parts[1] in valid_dom else "wait"
    r_int = parts[2] if len(parts) > 2 and parts[2] in valid_int else "wait"

    return dev, r_dom, r_int