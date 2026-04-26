from dataclasses import dataclass
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (from the paper)
# ─────────────────────────────────────────────────────────────────────────────

# Chinchilla scaling law constants (Hoffmann et al., 2022)
CHINCHILLA_E     = 1.69
CHINCHILLA_A     = 406.4
CHINCHILLA_B     = 410.0
CHINCHILLA_ALPHA = 0.34
CHINCHILLA_BETA  = 0.28

# Parameter growth: 3.7x per year = (3.7)^(1/52) per week
PARAM_GROWTH_ANNUAL = 3.7
PARAM_GROWTH_WEEKLY = PARAM_GROWTH_ANNUAL ** (1 / 52)

# Gompertz: planning ability doubles every 7 months (~30 weeks) in early phase
# Perfect planning = 80% success on 100-year tasks
# Current best (Claude Sonnet 3.7) = 80% success on 15-minute tasks
# Original paper calibration (100-year perfect planning horizon) gives S0 ≈ 4.25e-8,
# which is too small for a 260-week simulation window (a2 stays near 0 throughout).
# Recalibrated to S0 = 0.01 so that planning ability grows visibly over 5 years.
S0 = 0.01

# AI Potency weights and exponents — user-editable defaults (from the paper)
W1_DEFAULT     = 0.4
W2_DEFAULT     = 0.6
OMEGA1_DEFAULT = 2.0
OMEGA2_DEFAULT = 1.5

# Regulation step size per week
REG_STEP = 0.01

# Biorisk constants
# λ₀: base attack rate without AI — calibrated from the Global Terrorism Database
# (38 incidents over 50 years globally → 0.76/year → 0.76/52 per week)
LAMBDA0      = 0.76 / 52 # from CARMA's paper
BIORISK_THETA = 75.0   # AI potency threshold above which dangerous capabilities emerge (raised to match equation-derived p scale)
ALPHA_MAX    = 3.0     # max multiplier on number of attacks — arbitrary (set by user)
BETA_MAX     = 2.0     # max multiplier on severity — arbitrary (set by user)
DEATHS_BASE  = 5 / 38  # from CARMA's paper — expected deaths per incident without AI (from 2001 anthrax case)

# Gini inequality coefficient — empirical, from Bordot & Lorentz (2021)
GINI_COEFF = 1.47

# Normalisation: chance loss for 100,000-option forced choice
CHANCE_LOSS = np.log(100_000)

# Conversion factors for Chinchilla units
BYTES_PER_TOKEN = 2    # ~2 bytes per token (1 token ≈ 4 chars ≈ 0.75 word ≈ 2-4 bytes)
BYTES_PER_PB    = 1e15 # bytes in a petabyte (SI definition: 10^15 bytes)


# ─────────────────────────────────────────────────────────────────────────────
# COUNTRY CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryConfig:
    """
    Initial conditions and parameters for one country.
    Empirical values sourced from World Bank (2022), Epoch AI (2024),
    and Stanford HAI AI Index (2024) where available.
    Remaining parameters (p0, N0, D0, w1_data, w2_data) are illustrative.
    """
    name: str

    # AI potency initial value (1-100)
    # Illustrative — based on qualitative ranking from Epoch AI GPU share,
    # Stanford HAI AI Index, and frontier model quality by country.
    p0: float

    # Initial number of model parameters (in billions)
    # Illustrative — based on publicly known model sizes by country.
    N0_billions: float

    # Domestic data stock (in petabytes)
    # Illustrative — based on estimated internet content by country.
    D0_domestic_pb: float

    # Data access weights
    # Illustrative — based on internet openness and data policy (firewall, GDPR, etc.)
    w1_data: float   # how accessible global training data stock is domestically
    w2_data: float   # how accessible domestic training data stock is globally

    # Global data stock reference (shared across all countries)
    D_global_pb: float = 1_000_000.0   # ~1 exabyte — rough estimate of global text data online
    D_max_pb:    float = 5_000_000.0   # ~5 exabytes — rough estimate of global data ceiling

    # AI Potency equation weights (from the paper, user-editable)
    w1:     float = W1_DEFAULT
    w2:     float = W2_DEFAULT
    omega1: float = OMEGA1_DEFAULT
    omega2: float = OMEGA2_DEFAULT

    # Initial domestic regulation index [0, 1]
    # Interpreted as the regulatory brake on AI potency growth.
    # Not directly calibrated — no standardised empirical index exists yet.
    reg_dom_0: float = 0.1

    # Initial Gini coefficient — from World Bank (2022)
    gini_0: float = 0.35

    # Initial GDP per capita (USD) — from World Bank (2022)
    gdp_per_capita_0: float = 30_000.0

    # GDP growth rate per week (approximate)
    # (1 + 0.0005)^52 ≈ 1.026 → ~2.6% annual growth
    gdp_growth_weekly: float = 0.0005

    # Strategy for rule-based agent (placeholder for LLM agent)
    strategy: str = "balanced"

    # Country name for LLM prompt (used when LLM agent is attached)
    llm_prompt_country: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT COUNTRY CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COUNTRIES = [
    CountryConfig(
        name="USA",
        p0=75.0,
        N0_billions=1000.0,
        D0_domestic_pb=400_000,
        w1_data=0.9,    # near-full access to global data (English-dominant, no firewall)
        w2_data=0.9,    # domestic data widely accessible globally
        reg_dom_0=0.15,
        gini_0=0.417,           # World Bank 2022
        gdp_per_capita_0=76_657,  # World Bank 2022
        strategy="hawkish"
    ),
    CountryConfig(
        name="China",
        p0=65.0,
        N0_billions=500.0,
        D0_domestic_pb=300_000,
        w1_data=0.4,    # restricted by Great Firewall
        w2_data=0.3,    # domestic data less accessible abroad (legal + linguistic)
        reg_dom_0=0.2,
        gini_0=0.36,            # World Bank 2022
        gdp_per_capita_0=12_971,  # World Bank 2022
        strategy="hawkish"
    ),
    CountryConfig(
        name="EU",
        p0=50.0,
        N0_billions=200.0,
        D0_domestic_pb=200_000,
        w1_data=0.85,   # near-full access, slightly lower than USA
        w2_data=0.85,   # slightly lower than USA due to GDPR data transfer constraints
        reg_dom_0=0.4,
        gini_0=0.3135,          # World Bank 2022 (weighted average across member states)
        gdp_per_capita_0=41_609,  # World Bank 2022 (Euro area)
        strategy="dovish"
    ),
    CountryConfig(
        name="India",
        p0=35.0,
        N0_billions=50.0,
        D0_domestic_pb=100_000,
        w1_data=0.7,    # good internet access, growing but below Western levels
        w2_data=0.6,    # domestic data accessible but less so than Western countries
        reg_dom_0=0.1,
        gini_0=0.255,           # World Bank 2022
        gdp_per_capita_0=2_347,   # World Bank 2022
        strategy="balanced"
    ),
    CountryConfig(
        name="Russia",
        p0=40.0,
        N0_billions=80.0,
        D0_domestic_pb=80_000,
        w1_data=0.5,    # growing internet restrictions (RuNet), partial access
        w2_data=0.4,    # domestic data less accessible globally since 2022 sanctions
        reg_dom_0=0.1,
        gini_0=0.339,           # World Bank 2022
        gdp_per_capita_0=15_620,  # World Bank 2022
        strategy="hawkish"
    ),
]