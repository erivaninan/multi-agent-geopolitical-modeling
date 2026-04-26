import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random

from config import CountryConfig, DEFAULT_COUNTRIES
from equations import biorisk_attacks_per_week, biorisk_severity, REG_STEP, welfare
from country import CountryState
from agent import rule_based_action

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class Simulation:
    """
    Main simulation loop — 260 weekly rounds.
    Each round:
      1. All agents observe state and choose actions simultaneously.
      2. International regulation threshold is checked.
      3. All country states are updated.
      4. Global biorisk is computed.
    """

    def __init__(self,
                 countries: List[CountryConfig] = None,
                 n_weeks: int = 260,
                 seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        self.n_weeks  = n_weeks
        self.reg_int  = 0.1   # initial global regulation index

        # Instantiate country states with the correct initial reg_int
        configs = countries or DEFAULT_COUNTRIES
        self.countries = [
            CountryState(cfg, reg_int_init=self.reg_int) for cfg in configs
        ]

        # Global state history
        self.history_reg_int:         List[float] = []
        self.history_biorisk_attacks: List[float] = []
        self.history_biorisk_deaths:  List[float] = []
        self.history_p_max:           List[float] = []

    def _check_international_regulation(self, actions: Dict[str, Tuple]) -> str:
        """
        Threshold mechanism from paper Section 2.2.2.

        Condition A (regulation increases):
            US and China both push for regulation.
        Condition B (deregulation):
            US and China both push for deregulation.

        Fallback: if 60% or more of all countries push in one direction,
        that direction wins. The 60% threshold is arbitrary and chosen
        to require a broad consensus beyond a simple majority.
        """
        reg_int_actions = {name: act[2] for name, act in actions.items()}

        # Primary condition: US and China must both agree
        usa_action = reg_int_actions.get("USA",   "wait")
        chn_action = reg_int_actions.get("China", "wait")

        if usa_action == "regulate_int" and chn_action == "regulate_int":
            return "regulate"
        if usa_action == "deregulate_int" and chn_action == "deregulate_int":
            return "deregulate"

        # Fallback: qualified majority (60%) of all countries
        reg_count   = sum(1 for a in reg_int_actions.values() if a == "regulate_int")
        dereg_count = sum(1 for a in reg_int_actions.values() if a == "deregulate_int")

        if reg_count > len(self.countries) * 0.6:
            return "regulate"
        if dereg_count > len(self.countries) * 0.6:
            return "deregulate"

        return "wait"

    def run(self) -> pd.DataFrame:
        """
        Run the simulation and return a DataFrame with all tracked variables.
        Week 0 records the initial state before any actions are taken.
        Weeks 1-260 record the state after each round of actions.
        """
        records = []

        # ── Week 0: record initial state before any actions ───────────────
        p_max_init     = max(cs.p for cs in self.countries)
        attacks_init   = biorisk_attacks_per_week(p_max_init, self.reg_int)
        severity_init  = biorisk_severity(p_max_init)

        row0 = {
            "week":                      0,
            "reg_int":                   self.reg_int,
            "biorisk_attacks_per_week":  attacks_init,
            "biorisk_expected_deaths":   attacks_init * severity_init,
            "p_max_global":              p_max_init,
        }
        for cs in self.countries:
            n = cs.config.name
            row0[f"{n}_p"]          = cs.p
            row0[f"{n}_reg_dom"]    = cs.reg_dom
            row0[f"{n}_gini"]       = cs.gini
            row0[f"{n}_welfare"]    = welfare(cs.gini, cs.gdp_per_capita)
            row0[f"{n}_gdp"]        = cs.gdp_per_capita
            row0[f"{n}_action_dev"] = "none"
        records.append(row0)

        # ── Weeks 1-260: main simulation loop ─────────────────────────────
        for t in range(self.n_weeks):
            p_values = [cs.p for cs in self.countries]
            p_max    = max(p_values)

            # 1. Each agent chooses actions simultaneously
            actions = {}
            for cs in self.countries:
                p_rivals    = [p for p in p_values if p != cs.p]
                p_max_rival = max(p_rivals) if p_rivals else 0.0
                dev, r_dom, r_int = rule_based_action(
                    cs.config.strategy,
                    cs.p, cs.reg_dom, self.reg_int,
                    p_max_rival, t
                )
                actions[cs.config.name] = (dev, r_dom, r_int)

            # 2. Update international regulation
            reg_int_decision = self._check_international_regulation(actions)
            if reg_int_decision == "regulate":
                self.reg_int = min(1.0, self.reg_int + REG_STEP)
            elif reg_int_decision == "deregulate":
                self.reg_int = max(0.0, self.reg_int - REG_STEP)

            # 3. Update all country states
            for cs in self.countries:
                dev, r_dom, r_int = actions[cs.config.name]
                cs.step(dev, r_dom, self.reg_int, p_max)

            # 4. Recompute p_max after updates
            p_max_new = max(cs.p for cs in self.countries)

            # 5. Compute global biorisk
            attacks        = biorisk_attacks_per_week(p_max_new, self.reg_int)
            severity       = biorisk_severity(p_max_new)
            expected_deaths = attacks * severity

            # 6. Record global state history
            self.history_reg_int.append(self.reg_int)
            self.history_biorisk_attacks.append(attacks)
            self.history_biorisk_deaths.append(expected_deaths)
            self.history_p_max.append(p_max_new)

            # 7. Build record row
            row = {
                "week":                      t + 1,
                "reg_int":                   self.reg_int,
                "biorisk_attacks_per_week":  attacks,
                "biorisk_expected_deaths":   expected_deaths,
                "p_max_global":              p_max_new,
            }
            for cs in self.countries:
                n = cs.config.name
                row[f"{n}_p"]          = cs.p
                row[f"{n}_reg_dom"]    = cs.reg_dom
                row[f"{n}_gini"]       = cs.gini
                row[f"{n}_welfare"]    = welfare(cs.gini, cs.gdp_per_capita)
                row[f"{n}_gdp"]        = cs.gdp_per_capita
                row[f"{n}_action_dev"] = actions[n][0]
            records.append(row)

        return pd.DataFrame(records)