from config import DEFAULT_COUNTRIES
from simulation import Simulation
from plots import plot_results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running CARMA simulation (260 weeks, 5 countries and/or regions)...")
    sim = Simulation(countries=DEFAULT_COUNTRIES, n_weeks=260, seed=42)
    df  = sim.run()
    df.to_csv("results.csv", index=False)
    plot_results(df, DEFAULT_COUNTRIES, save_path="simulation.png")