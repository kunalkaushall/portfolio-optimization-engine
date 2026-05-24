"""
================================================================================
  MARKOWITZ MEAN-VARIANCE OPTIMIZATION  |  INSTITUTIONAL QUANT TERMINAL
  ───────────────────────────────────────────────────────────────────────
  Developed  : Kunal Kaushal
  Programme  : Advanced Financial Analytics — IIT Kanpur (NPTEL / Swayam)
  ───────────────────────────────────────────────────────────────────────
  Engine     : SciPy SLSQP  ·  Monte Carlo 5,000  ·  Historical VaR/CVaR
  Fallback   : Synthetic MVN (756 days)  ·  Tikhonov regularisation (lam=1e-6)
  Universe   : Nifty 50 + High-Momentum Pool (55 tickers, 12 sectors)
  Run        : streamlit run markowitz_dashboard_v2.py

  Architecture
  ────────────
  §1  Ticker Repair    — sanitise user-supplied symbols (.NS auto-append)
  §2  Data Layer       — live yfinance download <-> synthetic MVN fallback
  §3  Math Backend     — Tikhonov-regularised covariance, SLSQP solvers,
                         Monte Carlo simulation, Historical VaR / CVaR
  §4  Plotting Layer   — interactive Plotly dark-terminal efficient frontier
  §5  Streamlit UI     — 3-tab layout: Overview | Quant Terminal | Report

  Mathematical Foundations
  ────────────────────────
  Portfolio return     :  R_p  = w^T mu
  Portfolio variance   :  s2_p = w^T Sigma w
  Sharpe ratio         :  SR   = (R_p - R_f) / s_p
  Tikhonov (ridge) reg :  Sigma_reg = Sigma + lam*I,   lam = 1e-6
  SLSQP constraints    :  sum(w_i) = 1,  0 <= w_i <= 1  (long-only)
  VaR (95%)            :  5th percentile of historical portfolio daily returns
  CVaR (95%)           :  E[r | r <= VaR]  (expected shortfall)
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import plotly.graph_objects as go          # interactive chart engine
import streamlit as st

from scipy.optimize import minimize
from datetime       import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "Markowitz MVO | Quant Terminal",
    page_icon             = "◈",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  §A  NIFTY SECTOR POOL  — 55-ticker NSE universe
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_SECTOR_POOL: list[str] = [
    # Banking & Financial Services
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",     "AXISBANK.NS",  "KOTAKBANK.NS",
    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "IRFC.NS",     "JIOFIN.NS",
    # Information Technology
    "TCS.NS",      "INFY.NS",     "HCLTECH.NS",   "WIPRO.NS",     "TECHM.NS",
    # Energy & Oil/Gas
    "RELIANCE.NS", "ONGC.NS",     "BPCL.NS",      "NTPC.NS",      "POWERGRID.NS",
    "COALINDIA.NS","SUZLON.NS",
    # Automobiles
    "TATAMOTORS.NS","MARUTI.NS",  "M&M.NS",       "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    # Infrastructure & Capital Goods
    "LT.NS",       "ADANIENT.NS", "ADANIPORTS.NS","BEL.NS",       "HAL.NS",
    # Pharmaceuticals & Healthcare
    "SUNPHARMA.NS","DRREDDY.NS",  "CIPLA.NS",     "APOLLOHOSP.NS","DIVISLAB.NS",
    # FMCG & Consumer Staples
    "HINDUNILVR.NS","ITC.NS",     "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    # Consumer Discretionary
    "TITAN.NS",    "ASIANPAINT.NS","ZOMATO.NS",   "TRENT.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Cement & Materials
    "ULTRACEMCO.NS","GRASIM.NS",
    # Metals & Mining
    "TATASTEEL.NS","JSWSTEEL.NS", "HINDALCO.NS",
]

_SECTOR_MAP: dict[str, str] = {
    "HDFCBANK.NS"  :"Bank",    "ICICIBANK.NS" :"Bank",    "SBIN.NS"      :"Bank",
    "AXISBANK.NS"  :"Bank",    "KOTAKBANK.NS" :"Bank",    "INDUSINDBK.NS":"Bank",
    "BAJFINANCE.NS":"Finance",  "BAJAJFINSV.NS":"Finance", "HDFCLIFE.NS"  :"Insurance",
    "SBILIFE.NS"   :"Insurance","IRFC.NS"      :"Finance", "JIOFIN.NS"    :"Finance",
    "TCS.NS"       :"Tech",    "INFY.NS"      :"Tech",    "HCLTECH.NS"   :"Tech",
    "WIPRO.NS"     :"Tech",    "TECHM.NS"     :"Tech",
    "RELIANCE.NS"  :"Energy",  "ONGC.NS"      :"Energy",  "BPCL.NS"      :"Energy",
    "NTPC.NS"      :"Energy",  "POWERGRID.NS" :"Energy",  "COALINDIA.NS" :"Energy",
    "SUZLON.NS"    :"Energy",
    "TATAMOTORS.NS":"Auto",    "MARUTI.NS"    :"Auto",    "M&M.NS"       :"Auto",
    "EICHERMOT.NS" :"Auto",    "HEROMOTOCO.NS":"Auto",    "BAJAJ-AUTO.NS":"Auto",
    "LT.NS"        :"Infra",   "ADANIENT.NS"  :"Infra",   "ADANIPORTS.NS":"Infra",
    "BEL.NS"       :"Defence", "HAL.NS"       :"Defence",
    "SUNPHARMA.NS" :"Pharma",  "DRREDDY.NS"   :"Pharma",  "CIPLA.NS"     :"Pharma",
    "APOLLOHOSP.NS":"Pharma",  "DIVISLAB.NS"  :"Pharma",
    "HINDUNILVR.NS":"FMCG",    "ITC.NS"       :"FMCG",    "NESTLEIND.NS" :"FMCG",
    "BRITANNIA.NS" :"FMCG",    "TATACONSUM.NS":"FMCG",
    "TITAN.NS"     :"Consumer","ASIANPAINT.NS":"Consumer", "ZOMATO.NS"    :"Consumer",
    "TRENT.NS"     :"Consumer",
    "BHARTIARTL.NS":"Telecom",
    "ULTRACEMCO.NS":"Cement",  "GRASIM.NS"    :"Cement",
    "TATASTEEL.NS" :"Metal",   "JSWSTEEL.NS"  :"Metal",   "HINDALCO.NS"  :"Metal",
}

DEFAULT_TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
]

# ─────────────────────────────────────────────────────────────────────────────
#  §B  SIMULATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_OPTIONS: dict[str, int] = {
    "1 Day (Daily Check)": 7,
    "1 Month"            : 30,
    "3 Months"           : 90,
    "6 Months"           : 180,
    "1 Year"             : 365,
    "3 Years"            : 1_095,
    "5 Years"            : 1_825,
}
LOOKBACK_DEFAULT  = "3 Years"
_MIN_ROWS_FLOOR   = 5
_COV_REGULARISER  = 1e-6
DEFAULT_RF_PCT    = 5.0
TRADING_DAYS      = 252
N_SIM             = 5_000
RANDOM_SEED       = 42
SIM_YEARS         = 3
SIM_DAYS          = SIM_YEARS * TRADING_DAYS

_INDIAN_HINTS = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ZOMATO","TATASTEEL",
    "SUZLON","IRFC","JIOFIN","WIPRO","HINDUNILVR","BAJFINANCE","AXISBANK",
    "KOTAKBANK","LT","SBIN","SUNPHARMA","ONGC","NTPC","POWERGRID","ADANIENT",
    "ADANIPORTS","TITAN","ASIANPAINT","NESTLEIND","ULTRACEMCO","MARUTI",
    "BHARTIARTL","TECHM","HCLTECH","INDUSINDBK","JSWSTEEL","COALINDIA",
    "GRASIM","EICHERMOT","CIPLA","DRREDDY","DIVISLAB","APOLLOHOSP",
    "BRITANNIA","BPCL","HEROMOTOCO","TATACONSUM","SBILIFE","BAJAJFINSV",
    "HDFCLIFE","M&M","TATAMOTORS","TATAPOWER","HAL","BEL","TRENT","HINDALCO",
    "TATASTEEL","BAJAJ-AUTO",
}

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — institutional dark terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── canvas ───────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background:#08080f; }
[data-testid="stSidebar"]          { background:#0c0c18;
                                     border-right:1px solid #18183a; }
[data-testid="stHeader"]           { background:#08080f; }

/* ── scrollbars ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child { overflow-y:auto; height:100vh; }
[data-testid="stAppViewContainer"] > section:nth-child(2) {
    overflow-y:auto; height:100vh;
}
::-webkit-scrollbar              { width:5px; height:5px; }
::-webkit-scrollbar-track        { background:#08080f; }
::-webkit-scrollbar-thumb        { background:#1e1e40; border-radius:4px; }
::-webkit-scrollbar-thumb:hover  { background:#2a2a60; }

/* ── tabs ─────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #1a1a3a !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    background    : transparent !important;
    border-bottom : 2px solid transparent !important;
    color         : #555580 !important;
    font-size     : 0.82rem !important;
    font-weight   : 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding       : 10px 22px !important;
    transition    : all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color         : #9090cc !important;
    border-bottom : 2px solid #3a3a80 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color         : #c0c8ff !important;
    border-bottom : 2px solid #5555cc !important;
    background    : #0e0e20 !important;
}

/* ── metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background    : #0e0e1e;
    border        : 1px solid #1e1e40;
    border-radius : 10px;
    padding       : 14px 18px;
    box-shadow    : 0 2px 10px rgba(20,20,70,0.3);
    transition    : all 0.3s cubic-bezier(0.4,0,0.2,1);
    position      : relative;
    overflow      : hidden;
}
[data-testid="metric-container"]::before {
    content    : '';
    position   : absolute;
    top:0; left:0; right:0; height:2px;
    background : linear-gradient(90deg, #2a2a88, #5555cc, #00BFFF);
    opacity    : 0;
    transition : opacity 0.3s ease;
}
[data-testid="metric-container"]:hover {
    border-color : #2e2e70;
    box-shadow   : 0 4px 20px rgba(50,50,140,0.35);
    transform    : translateY(-1px);
}
[data-testid="metric-container"]:hover::before { opacity:1; }
[data-testid="metric-container"] label {
    color         : #5a5a8a !important;
    font-size     : 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color        : #dde0f8;
    font-size    : 1.18rem;
    font-weight  : 700;
    letter-spacing: 0.01em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem;
}

/* ── multiselect & inputs ─────────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div {
    background  : #0e0e1e !important;
    border      : 1px solid #20205a !important;
    border-radius: 8px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background  : #1a1a55 !important;
    border      : 1px solid #2e2e80 !important;
    border-radius: 5px !important;
    color       : #b0b4ff !important;
    font-size   : 0.70rem !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg { fill:#7070bb !important; }
[data-testid="stSelectbox"] > div > div {
    background  : #0e0e1e !important;
    border      : 1px solid #20205a !important;
    border-radius: 8px !important;
    color       : #b0b4ff !important;
}
[data-testid="stTextInput"] input {
    background  : #0e0e1e !important;
    border      : 1px solid #20205a !important;
    border-radius: 8px !important;
    color       : #d0d4ff !important;
}
[data-testid="stExpander"] {
    background  : #0c0c1e !important;
    border      : 1px solid #18183a !important;
    border-radius: 8px !important;
}
[data-testid="stDataFrame"] {
    border      : 1px solid #1e1e40;
    border-radius: 8px;
}

/* ── typography ───────────────────────────────────────────────────────── */
h1 { color:#c0c8ff !important; font-size:1.7rem !important; letter-spacing:-0.01em !important; }
h2 { color:#8888cc !important; font-size:1.15rem !important; letter-spacing:0.04em !important;
     text-transform:uppercase !important; }
h3 { color:#6868aa !important; font-size:1.0rem !important; }
p, li, label, span, div { color:#bbbdd6 !important; }
hr { border-color:#141430; }
code { color:#88ccff !important; background:#0e0e28 !important; }

/* ── run button ───────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background  : linear-gradient(135deg,#252580,#3838aa);
    color       : #ffffff;
    border      : 1px solid #3a3a88;
    border-radius: 8px;
    font-weight : 600;
    padding     : 10px 0;
    width       : 100%;
    letter-spacing: 0.05em;
    font-size   : 0.82rem;
    transition  : all .2s;
}
[data-testid="stButton"] > button:hover {
    background  : linear-gradient(135deg,#303090,#4545bb);
    box-shadow  : 0 0 18px rgba(60,60,180,0.35);
}

/* ── sector pills ─────────────────────────────────────────────────────── */
.sector-pill {
    display      : inline-block;
    background   : #14143a;
    border       : 1px solid #28286a;
    border-radius: 3px;
    padding      : 1px 6px;
    font-size    : 0.67rem;
    color        : #8888cc !important;
    margin-right : 3px;
    margin-bottom: 3px;
    letter-spacing: 0.04em;
}

/* ── lookback warning ─────────────────────────────────────────────────── */
.lookback-warn {
    background  : #18110a;
    border      : 1px solid #443300;
    border-radius: 6px;
    padding     : 6px 10px;
    font-size   : 0.73rem;
    color       : #bbaa44 !important;
    margin-top  : 4px;
}

/* ── capital allocation panel ─────────────────────────────────────────── */
.alloc-panel {
    background    : #0c0c1e;
    border        : 1px solid #1e1e44;
    border-radius : 10px;
    padding       : 18px 22px;
}
.alloc-header-sharpe {
    font-size     : 0.72rem;
    font-weight   : 700;
    letter-spacing: 0.1em;
    color         : #EF5350 !important;
    text-transform: uppercase;
    border-bottom : 1px solid #2a1a1a;
    padding-bottom: 6px;
    margin-bottom : 12px;
}
.alloc-header-minvar {
    font-size     : 0.72rem;
    font-weight   : 700;
    letter-spacing: 0.1em;
    color         : #26A69A !important;
    text-transform: uppercase;
    border-bottom : 1px solid #1a2a2a;
    padding-bottom: 6px;
    margin-bottom : 12px;
}
.overview-block {
    background    : #0c0c1e;
    border        : 1px solid #1e1e44;
    border-left   : 3px solid #3a3aaa;
    border-radius : 0 8px 8px 0;
    padding       : 18px 24px;
    margin-bottom : 18px;
    line-height   : 1.75;
}
.report-card {
    background    : #0c0c1e;
    border        : 1px solid #1e1e44;
    border-radius : 10px;
    padding       : 20px 26px;
    margin-bottom : 18px;
}
.report-card h4 {
    color         : #8888cc !important;
    font-size     : 0.68rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom : 10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  §1  TICKER REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def _repair_tickers(raw: str) -> list[str]:
    """Parse free-text ticker input and return a cleaned, deduplicated list.
    Bare Indian names (e.g. WIPRO) become WIPRO.NS automatically."""
    tokens = [
        t.strip().upper()
        for t in raw.replace(",", "\n").splitlines()
        if t.strip()
    ]
    seen:   dict[str, None] = {}
    result: list[str]       = []
    for t in tokens:
        if t not in seen:
            seen[t] = None
            result.append(t if "." in t else t + ".NS")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  §2  DATA LAYER  — live download with synthetic fallback
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3_600)
def _yf_download(tickers: tuple, lookback_days: int) -> pd.DataFrame:
    """1-hour cached yfinance download. Returns raw close-price DataFrame."""
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    raw   = yf.download(
        list(tickers),
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False,
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    raw = raw.reindex(columns=list(tickers))
    return raw


def _synthetic_prices(
    tickers : list[str],
    n_days  : int = SIM_DAYS,
    seed    : int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic price DataFrame via multivariate normal log-return
    simulation — offline-resilient fallback engine.

    Methodology:
    1. Draw mu_i ~ U(8%,28%) and sigma_i ~ U(12%,38%) per asset.
    2. One-factor correlation: C_ij = beta_i * beta_j * 0.5 (off-diagonal).
    3. Convert correlation to covariance: Sigma = D * C * D.
    4. Cholesky decomposition of daily Sigma.
    5. Simulate: r_t = mu/252 + L * z_t, z_t ~ N(0,I).
    6. Cumulative exponentiation -> price paths anchored at 100.
    """
    rng = np.random.default_rng(seed)
    n   = len(tickers)

    ann_rets     = rng.uniform(0.08, 0.28, n)
    ann_vols     = rng.uniform(0.12, 0.38, n)
    factor_loads = rng.uniform(0.3,  0.8,  n)

    corr = np.outer(factor_loads, factor_loads) * 0.5
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -0.99, 0.99)

    D          = np.diag(ann_vols)
    cov        = D @ corr @ D
    daily_mu   = ann_rets / TRADING_DAYS
    daily_cov  = cov      / TRADING_DAYS

    L  = np.linalg.cholesky(daily_cov + 1e-8 * np.eye(n))
    z  = rng.standard_normal((n_days, n))
    lr = daily_mu + (L @ z.T).T
    prices = np.exp(np.cumsum(lr, axis=0)) * 100.0

    bdays = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame(prices, index=bdays, columns=tickers)


def load_price_data(
    tickers      : list[str],
    lookback_days: int,
    lookback_label: str,
) -> tuple[pd.DataFrame, bool, str]:
    """Master data-loading: attempts live yfinance first, falls back to
    synthetic MVN on any failure or insufficient rows.

    CRITICAL dropna guard: raw.dropna(axis=1, how='all').dropna() removes
    all-NaN ticker columns first, then drops any rows with remaining NaN,
    guaranteeing a clean rectangular matrix before any covariance math.
    """
    n_assets = len(tickers)
    min_rows = max(n_assets + 2, _MIN_ROWS_FLOOR)

    try:
        raw = _yf_download(tuple(tickers), lookback_days)
        # CRITICAL: drop all-NaN columns first, then drop any remaining NaN rows
        raw = raw.dropna(axis=1, how="all").dropna()

        if raw.shape[1] < 2:
            raise ValueError(
                f"Fewer than 2 tickers returned data for '{lookback_label}'."
            )
        if raw.shape[0] < min_rows:
            raise ValueError(
                f"Only {raw.shape[0]} trading day(s) for '{lookback_label}' "
                f"— need >= {min_rows} for a valid {n_assets}-asset covariance matrix."
            )
        valid = [t for t in tickers if t in raw.columns]
        return raw[valid], False, ""

    except Exception as exc:
        reason = str(exc)
        st.sidebar.warning(
            f"**Live data unavailable:**\n`{reason}`\n\n"
            "**Switching to Synthetic MVN Engine** — 756 simulated trading "
            "days. All optimisation math is identical; only the price series "
            "is artificial."
        )
        synth = _synthetic_prices(tickers, n_days=SIM_DAYS, seed=RANDOM_SEED)
        return synth, True, reason


# ══════════════════════════════════════════════════════════════════════════════
#  §3  MATHEMATICAL BACKEND  — pure functions, no Streamlit dependencies
# ══════════════════════════════════════════════════════════════════════════════

def log_returns_df(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily logarithmic returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def regularize_cov(cov: np.ndarray, lam: float = _COV_REGULARISER) -> np.ndarray:
    """Tikhonov (ridge) regularisation: Sigma_reg = Sigma + lam * I.
    Guarantees positive-definiteness for any lam > 0."""
    return cov + lam * np.eye(cov.shape[0])


def annualize_stats(lr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualise daily log-return statistics.
    mu_annual = E[r_daily] * 252 | Sigma_annual = Cov(r_daily) * 252
    Applies Tikhonov regularisation to annualised covariance."""
    mu  = lr.mean().values * TRADING_DAYS
    cov = regularize_cov(lr.cov().values * TRADING_DAYS)
    return mu, cov


def p_ret(w: np.ndarray, mu: np.ndarray) -> float:
    """Portfolio annualised expected return: R_p = w^T * mu."""
    return float(w @ mu)


def p_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio annualised volatility: s_p = sqrt(w^T * Sigma * w)."""
    v = float(w @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))


def p_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    """Sharpe ratio: SR = (R_p - R_f) / s_p. Returns 0 if vol ~ 0."""
    vol = p_vol(w, cov)
    return (p_ret(w, mu) - rf) / vol if vol > 1e-12 else 0.0


def _cb(n: int):
    """Build SLSQP equality constraint and box bounds for n-asset portfolio."""
    return (
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        tuple((0.0, 1.0) for _ in range(n)),
    )


def opt_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Maximum Sharpe Ratio portfolio via SciPy SLSQP.
    Objective: min_w -SR(w) = -(w^T mu - R_f) / sqrt(w^T Sigma w)
    Falls back to equal-weight if optimiser diverges."""
    n    = len(mu)
    w0   = np.full(n, 1.0 / n)
    con, bnd = _cb(n)
    try:
        res = minimize(
            lambda w: -p_sharpe(w, mu, cov, rf),
            w0, method="SLSQP", bounds=bnd, constraints=con,
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        if not res.success:
            raise RuntimeError(res.message)
        w = np.clip(res.x, 0, 1)
        w /= w.sum()
        return w
    except Exception:
        return w0


def opt_min_var(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Global Minimum Variance portfolio via SciPy SLSQP.
    Objective: min_w s_p(w) = sqrt(w^T Sigma w)
    Falls back to equal-weight if optimiser diverges."""
    n    = len(mu)
    w0   = np.full(n, 1.0 / n)
    con, bnd = _cb(n)
    try:
        res = minimize(
            lambda w: p_vol(w, cov),
            w0, method="SLSQP", bounds=bnd, constraints=con,
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        if not res.success:
            raise RuntimeError(res.message)
        w = np.clip(res.x, 0, 1)
        w /= w.sum()
        return w
    except Exception:
        return w0


def monte_carlo(
    mu: np.ndarray, cov: np.ndarray, rf: float,
    n: int = N_SIM, seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate n random portfolios to visualise the feasible set.
    Each portfolio is Dirichlet-like: w_i ~ Uniform(0,1)^N, normalised to sum=1."""
    rng = np.random.default_rng(seed)
    N   = len(mu)
    R, V, S = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        w    = rng.random(N); w /= w.sum()
        R[i] = p_ret(w, mu)
        V[i] = p_vol(w, cov)
        S[i] = p_sharpe(w, mu, cov, rf)
    return R, V, S


def asset_metrics(
    mu: np.ndarray, cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Annualised return and volatility for each individual asset (100% allocation)."""
    n = len(mu)
    rets, vols = np.empty(n), np.empty(n)
    for i in range(n):
        w = np.zeros(n); w[i] = 1.0
        rets[i] = p_ret(w, mu)
        vols[i] = p_vol(w, cov)
    return rets, vols


def hist_var_cvar(
    w: np.ndarray, lr: pd.DataFrame, conf: float = 0.95,
) -> tuple[float, float]:
    """95% Historical Value-at-Risk and Conditional VaR (Expected Shortfall).
    VaR  = 5th percentile of portfolio daily return distribution.
    CVaR = E[r_p | r_p <= VaR] — mean of all returns in the loss tail."""
    pret  = lr.values @ w
    alpha = 1.0 - conf
    var   = float(np.percentile(pret, alpha * 100))
    tail  = pret[pret <= var]
    cvar  = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


# ══════════════════════════════════════════════════════════════════════════════
#  §4  PLOTTING LAYER — institutional dark-terminal efficient frontier
# ══════════════════════════════════════════════════════════════════════════════

_PALETTE = [
    "#4FC3F7","#FFD54F","#F48FB1","#FFAB40","#69F0AE",
    "#CE93D8","#4DD0E1","#FF7043","#D4E157","#80DEEA",
    "#FF8A65","#9575CD","#4DB6AC","#FFF176","#F06292",
    "#A5D6A7","#FFB74D","#B3E5FC","#FF8A80","#82B1FF",
    "#CCFF90","#EA80FC","#84FFFF","#FF6E40","#40C4FF",
    "#B9F6CA","#FF4081","#EEFF41","#7C4DFF","#64FFDA",
    "#FFD180","#FFFFFF","#AEEA00","#FF6D00","#00B0FF",
]

# Nudge table: ticker_name -> (Δvol_fraction, Δret_fraction, horizontal_align)
_NUDGE: dict[str, tuple] = {
    "RELIANCE"   :(+0.030,+0.075,"left"),  "TCS"        :(-0.080,+0.075,"right"),
    "INFY"       :(+0.030,-0.090,"left"),  "HDFCBANK"   :(-0.090,+0.075,"right"),
    "ICICIBANK"  :(+0.030,+0.075,"left"),  "SBIN"       :(-0.060,-0.090,"right"),
    "AXISBANK"   :(+0.030,-0.090,"left"),  "KOTAKBANK"  :(-0.090,+0.075,"right"),
    "BAJFINANCE" :(+0.030,+0.075,"left"),  "BAJAJFINSV" :(-0.095,-0.090,"right"),
    "INDUSINDBK" :(+0.030,+0.075,"left"),  "HDFCLIFE"   :(-0.090,-0.090,"right"),
    "SBILIFE"    :(+0.030,-0.090,"left"),  "IRFC"       :(-0.060,+0.075,"right"),
    "JIOFIN"     :(+0.030,+0.075,"left"),  "HCLTECH"    :(-0.085,-0.090,"right"),
    "WIPRO"      :(+0.030,+0.075,"left"),  "TECHM"      :(-0.075,+0.075,"right"),
    "ONGC"       :(+0.030,-0.090,"left"),  "BPCL"       :(-0.060,+0.075,"right"),
    "NTPC"       :(+0.030,+0.075,"left"),  "POWERGRID"  :(-0.095,-0.090,"right"),
    "COALINDIA"  :(+0.030,-0.090,"left"),  "SUZLON"     :(-0.070,+0.075,"right"),
    "TATAMOTORS" :(+0.030,+0.075,"left"),  "MARUTI"     :(-0.075,-0.090,"right"),
    "M&M"        :(+0.030,-0.090,"left"),  "EICHERMOT"  :(-0.090,+0.075,"right"),
    "HEROMOTOCO" :(+0.030,+0.075,"left"),  "BAJAJ-AUTO" :(-0.095,-0.090,"right"),
    "ADANIENT"   :(+0.030,+0.075,"left"),  "ADANIPORTS" :(+0.030,-0.090,"left"),
    "BEL"        :(-0.050,+0.075,"right"), "HAL"        :(+0.030,+0.075,"left"),
    "SUNPHARMA"  :(-0.095,+0.075,"right"), "DRREDDY"    :(+0.030,+0.075,"left"),
    "CIPLA"      :(-0.060,-0.090,"right"), "APOLLOHOSP" :(+0.030,-0.090,"left"),
    "DIVISLAB"   :(-0.075,+0.075,"right"), "HINDUNILVR" :(+0.030,+0.075,"left"),
    "ITC"        :(-0.045,-0.090,"right"), "NESTLEIND"  :(-0.085,-0.090,"right"),
    "BRITANNIA"  :(+0.030,+0.075,"left"),  "TATACONSUM" :(-0.090,+0.075,"right"),
    "TITAN"      :(+0.030,-0.090,"left"),  "ASIANPAINT" :(-0.090,+0.075,"right"),
    "ZOMATO"     :(+0.030,+0.075,"left"),  "TRENT"      :(+0.030,+0.075,"left"),
    "BHARTIARTL" :(-0.095,+0.075,"right"), "ULTRACEMCO" :(-0.095,-0.090,"right"),
    "GRASIM"     :(+0.030,+0.075,"left"),  "TATASTEEL"  :(-0.085,+0.075,"right"),
    "JSWSTEEL"   :(+0.030,-0.090,"left"),  "HINDALCO"   :(-0.080,+0.075,"right"),
    "LT"         :(+0.030,+0.075,"left"),
}


def build_figure(
    mu            : np.ndarray,
    cov           : np.ndarray,
    ws            : np.ndarray,
    wm            : np.ndarray,
    mc_r          : np.ndarray,
    mc_v          : np.ndarray,
    mc_s          : np.ndarray,
    lr            : pd.DataFrame,
    names         : list[str],
    rf            : float,
    is_synth      : bool = False,
    lookback_label: str  = "3 Years",
) -> go.Figure:
    """
    Interactive Efficient Frontier chart built with Plotly graph_objects.

    Design principles (matching the original institutional dark-terminal look):
    - plot_bgcolor / paper_bgcolor  = #08080f  (deep navy-black)
    - Gridlines at #101025 (barely visible — financial terminal style)
    - Monte Carlo cloud coloured by Sharpe ratio via plasma colorscale with colorbar
    - Individual asset dots — per-ticker palette colour, full hover tooltip
    - Max Sharpe  = filled circle (crimson #EF5350)  with dashed crosshairs
    - Min Variance = diamond marker (teal #26A69A)   with dashed crosshairs
    - Risk-free rate = dotted horizontal reference line
    - Monospaced, uppercase title that changes colour when synthetic data is used

    Hover tooltips
    ──────────────
    Monte Carlo dots   : Return (%), Volatility (%), Sharpe Ratio
    Individual assets  : Ticker, Annual Return (%), Annual Volatility (%)
    Max Sharpe marker  : Return (%), Volatility (%), Sharpe Ratio
    Min Variance marker: Return (%), Volatility (%), Sharpe Ratio
    """
    rs, vs, ss = p_ret(ws, mu), p_vol(ws, cov), p_sharpe(ws, mu, cov, rf)
    rm, vm, sm = p_ret(wm, mu), p_vol(wm, cov), p_sharpe(wm, mu, cov, rf)
    ar, av     = asset_metrics(mu, cov)

    # ── axis range helpers ───────────────────────────────────────────────────
    all_v = np.concatenate([mc_v*100, av*100, [vs*100, vm*100]])
    all_r = np.concatenate([mc_r*100, ar*100, [rs*100, rm*100]])
    vlo, vhi = float(all_v.min()), float(all_v.max())
    rlo, rhi = float(all_r.min()), float(all_r.max())
    dv = max(vhi - vlo, 1e-4)
    dr = max(rhi - rlo, 1e-4)
    x_range = [vlo - 0.04*dv, vhi + 0.08*dv]
    y_range = [rlo - 0.06*dr, rhi + 0.12*dr]

    fig = go.Figure()

    # ── 1.  Monte Carlo feasible set — WebGL scatter, plasma Sharpe colormap ─
    fig.add_trace(go.Scattergl(
        x    = mc_v * 100,
        y    = mc_r * 100,
        mode = "markers",
        name = "Monte Carlo Portfolios",
        marker = dict(
            color      = mc_s,
            colorscale = "plasma",
            opacity    = 0.30,
            size       = 3,
            colorbar   = dict(
                title      = dict(
                    text     = "Sharpe Ratio",
                    font     = dict(color="#666688", size=10, family="monospace"),
                    side     = "right",
                ),
                tickfont   = dict(color="#666688", size=8, family="monospace"),
                outlinecolor = "#1a1a3a",
                outlinewidth = 1,
                thickness  = 14,
                len        = 0.75,
                x          = 1.01,
            ),
            showscale  = True,
            line       = dict(width=0),
        ),
        hovertemplate = (
            "<b>Monte Carlo Portfolio</b><br>"
            "Return : %{y:.2f}%<br>"
            "Volatility : %{x:.2f}%<br>"
            "Sharpe : %{marker.color:.3f}"
            "<extra></extra>"
        ),
        showlegend = False,
    ))

    # ── 2.  Individual asset dots — one trace per ticker ─────────────────────
    for i, name in enumerate(names):
        colour = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x    = [av[i] * 100],
            y    = [ar[i] * 100],
            mode = "markers+text",
            name = name,
            marker = dict(
                color  = colour,
                size   = 9,
                opacity = 0.90,
                line   = dict(color="rgba(255,255,255,0.12)", width=0.8),
            ),
            text          = [name],
            textposition  = "top center",
            textfont      = dict(color=colour, size=8, family="monospace"),
            hovertemplate = (
                f"<b>{name}</b><br>"
                "Return : %{y:.2f}%<br>"
                "Volatility : %{x:.2f}%"
                "<extra></extra>"
            ),
            showlegend = True,
        ))

    # ── 3.  Risk-free rate reference line (layout shape + annotation) ─────────
    fig.add_shape(
        type      = "line",
        x0        = x_range[0], x1 = x_range[1],
        y0        = rf * 100,   y1 = rf * 100,
        line      = dict(color="#333355", width=1.0, dash="dot"),
        layer     = "below",
    )
    fig.add_annotation(
        x         = x_range[0] + 0.015 * dv,
        y         = rf * 100 + 0.025 * dr,
        text      = f"Rf = {rf*100:.2f}%",
        showarrow = False,
        font      = dict(color="#444466", size=9, family="monospace"),
        xanchor   = "left",
    )

    # ── 4.  Max Sharpe — dashed crosshairs ───────────────────────────────────
    for shape_cfg in [
        dict(x0=vs*100, x1=vs*100, y0=y_range[0], y1=rs*100),   # vertical
        dict(x0=x_range[0], x1=vs*100, y0=rs*100, y1=rs*100),   # horizontal
    ]:
        fig.add_shape(
            type  = "line",
            line  = dict(color="#C62828", width=0.9, dash="dash"),
            layer = "below",
            **shape_cfg,
        )

    # ── 5.  Max Sharpe marker ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = [vs * 100],
        y    = [rs * 100],
        mode = "markers",
        name = f"MAX SHARPE   SR={ss:.3f}",
        marker = dict(
            symbol = "circle",
            color  = "#EF5350",
            size   = 16,
            line   = dict(color="#FFFFFF", width=1.5),
        ),
        hovertemplate = (
            "<b>Max Sharpe Portfolio</b><br>"
            "Return : %{y:.2f}%<br>"
            f"Volatility : {vs*100:.2f}%<br>"
            f"Sharpe : {ss:.3f}"
            "<extra></extra>"
        ),
        showlegend = True,
    ))

    # ── 6.  Min Variance — dashed crosshairs ─────────────────────────────────
    for shape_cfg in [
        dict(x0=vm*100, x1=vm*100, y0=y_range[0], y1=rm*100),
        dict(x0=x_range[0], x1=vm*100, y0=rm*100, y1=rm*100),
    ]:
        fig.add_shape(
            type  = "line",
            line  = dict(color="#00695C", width=0.9, dash="dash"),
            layer = "below",
            **shape_cfg,
        )

    # ── 7.  Min Variance marker ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = [vm * 100],
        y    = [rm * 100],
        mode = "markers",
        name = f"MIN VARIANCE   SR={sm:.3f}",
        marker = dict(
            symbol = "diamond",
            color  = "#26A69A",
            size   = 14,
            line   = dict(color="#FFFFFF", width=1.5),
        ),
        hovertemplate = (
            "<b>Min Variance Portfolio</b><br>"
            "Return : %{y:.2f}%<br>"
            f"Volatility : {vm*100:.2f}%<br>"
            f"Sharpe : {sm:.3f}"
            "<extra></extra>"
        ),
        showlegend = True,
    ))

    # ── 8.  Layout — dark terminal aesthetic ──────────────────────────────────
    if is_synth:
        title_text  = (
            f"MARKOWITZ MEAN-VARIANCE EFFICIENT FRONTIER  |  "
            f"{len(names)} ASSETS  |  SYNTHETIC MVN DATA [{lookback_label} REQUESTED]"
            f"  |  {N_SIM:,} MC PATHS"
        )
        title_colour = "#D4AA00"
    else:
        title_text  = (
            f"MARKOWITZ MEAN-VARIANCE EFFICIENT FRONTIER  |  "
            f"{len(names)} ASSETS  |  LIVE DATA  LOOKBACK: {lookback_label.upper()}"
            f"  |  {N_SIM:,} MC PATHS"
        )
        title_colour = "#555578"

    fig.update_layout(
        # ── canvas ──────────────────────────────────────────────────────────
        plot_bgcolor  = "#08080f",
        paper_bgcolor = "#08080f",
        height        = 640,
        margin        = dict(l=70, r=90, t=80, b=70),

        # ── title ────────────────────────────────────────────────────────────
        title = dict(
            text      = title_text,
            font      = dict(color=title_colour, size=11, family="monospace"),
            x         = 0.0,
            xanchor   = "left",
            pad       = dict(l=4, b=8),
        ),

        # ── axes ─────────────────────────────────────────────────────────────
        xaxis = dict(
            title      = dict(
                text   = "ANNUALISED VOLATILITY (%)",
                font   = dict(color="#555578", size=10, family="monospace"),
                standoff = 10,
            ),
            tickfont   = dict(color="#555578", size=8, family="monospace"),
            gridcolor  = "#101025",
            gridwidth  = 0.6,
            zerolinecolor = "#1a1a3a",
            linecolor  = "#141430",
            range      = x_range,
            showgrid   = True,
        ),
        yaxis = dict(
            title      = dict(
                text   = "ANNUALISED RETURN (%)",
                font   = dict(color="#555578", size=10, family="monospace"),
                standoff = 10,
            ),
            tickfont   = dict(color="#555578", size=8, family="monospace"),
            gridcolor  = "#101025",
            gridwidth  = 0.6,
            zerolinecolor = "#1a1a3a",
            linecolor  = "#141430",
            range      = y_range,
            showgrid   = True,
        ),

        # ── legend ────────────────────────────────────────────────────────────
        legend = dict(
            bgcolor     = "rgba(8,8,15,0.75)",
            bordercolor = "#1a1a3a",
            borderwidth = 1,
            font        = dict(color="#aaaacc", size=9, family="monospace"),
            x           = 0.01,
            y           = 0.01,
            xanchor     = "left",
            yanchor     = "bottom",
            itemsizing  = "constant",
        ),

        # ── hover ─────────────────────────────────────────────────────────────
        hoverlabel = dict(
            bgcolor   = "#0e0e1e",
            bordercolor = "#2a2a5a",
            font      = dict(color="#d0d4ff", size=11, family="monospace"),
        ),
        hovermode = "closest",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  §5  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  5.1  SIDEBAR — inputs, lookback, risk-free rate, capital slider, run button
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    # Asset Universe
    st.markdown("**Asset Universe**")
    st.caption(
        f"Select from {len(NIFTY_SECTOR_POOL)} NSE tickers across 12 sectors. "
        "All symbols carry the `.NS` suffix — no typing needed."
    )
    selected_tickers: list[str] = st.multiselect(
        label            = "Select Assets",
        options          = NIFTY_SECTOR_POOL,
        default          = DEFAULT_TICKERS,
        help             = (
            f"Choose 2–{len(NIFTY_SECTOR_POOL)} stocks. "
            "Feeds directly into the SLSQP optimiser, Monte Carlo engine, and VaR/CVaR."
        ),
        label_visibility = "collapsed",
    )

    if selected_tickers:
        sectors    = sorted({_SECTOR_MAP.get(t, "Other") for t in selected_tickers})
        badge_html = " ".join(
            f'<span class="sector-pill">{s}</span>' for s in sectors
        )
        st.markdown(
            f"<div style='margin-top:4px;margin-bottom:2px;'>{badge_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"**{len(selected_tickers)} asset"
            f"{'s' if len(selected_tickers)!=1 else ''} selected** "
            f"across {len(sectors)} sector{'s' if len(sectors)!=1 else ''}"
        )

    with st.expander("Add custom tickers (optional)"):
        st.caption(
            "One per line or comma-separated.  \n"
            "Bare Indian names (e.g. `WIPRO`) auto-become `WIPRO.NS`.  \n"
            "Non-NSE symbols (e.g. `AAPL`) pass through unchanged."
        )
        custom_raw = st.text_area(
            label            = "Custom tickers",
            value            = "",
            height           = 80,
            placeholder      = "e.g. WIPRO\nAAPL\nMSFT.US",
            label_visibility = "collapsed",
        )
        custom_tickers = _repair_tickers(custom_raw) if custom_raw.strip() else []
        if custom_tickers:
            st.caption("**Parsed:** " + "  ·  ".join(custom_tickers))

    # Deduplicate & merge
    seen_m: dict[str, None] = {}
    tickers: list[str]      = []
    for t in (selected_tickers + custom_tickers):
        if t not in seen_m:
            seen_m[t] = None
            tickers.append(t)

    st.markdown("---")

    # Historical Lookback Period
    st.markdown("**Historical Lookback Period**")
    lookback_label: str = st.selectbox(
        label            = "Lookback Period",
        options          = list(LOOKBACK_OPTIONS.keys()),
        index            = list(LOOKBACK_OPTIONS.keys()).index(LOOKBACK_DEFAULT),
        help             = (
            "Controls the date range sent to yfinance. "
            "Short windows (<= 1 Month) almost always trigger the Synthetic MVN "
            "fallback — too few trading days for a valid covariance matrix."
        ),
        label_visibility = "collapsed",
    )
    lookback_days: int = LOOKBACK_OPTIONS[lookback_label]

    if lookback_days <= 30:
        st.markdown(
            "<div class='lookback-warn'>"
            "<b>Short window selected.</b>  Fewer trading days than assets will "
            "activate the Synthetic MVN engine automatically."
            "</div>",
            unsafe_allow_html=True,
        )
    elif lookback_days <= 90:
        st.markdown(
            "<div class='lookback-warn'>"
            "<b>Short-medium window.</b>  Results are annualised from a limited "
            "sample — interpret with care."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Risk-Free Rate
    st.markdown("**Annual Risk-Free Rate (%)**")
    rf_pct: float = st.slider(
        label            = "RF (%)",
        min_value        = 0.0,
        max_value        = 15.0,
        value            = DEFAULT_RF_PCT,
        step             = 0.25,
        format           = "%.2f%%",
        label_visibility = "collapsed",
    )
    st.caption(f"Selected: **{rf_pct:.2f}%**  (Indian T-Bill proxy)")
    rf = rf_pct / 100.0

    st.markdown("---")

    # Total Capital Slider (NEW)
    st.markdown("**Total Capital to Invest (Rs.)**")
    total_capital: float = float(
        st.slider(
            label       = "Capital",
            min_value   = 1_000,
            max_value   = 100_000,
            value       = 100_000,
            step        = 1_000,
            format      = "Rs. %d",
            label_visibility = "collapsed",
            help        = "Used to compute per-stock rupee allocations in the "
                          "Capital Allocation Summary and Plain-English Report.",
        )
    )
    st.caption(
        f"Capital: **Rs. {total_capital:,.0f}**  "
        f"(Rs. {total_capital/1e3:.1f}K)"
    )

    st.markdown("---")
    run = st.button("Run Optimization", use_container_width=True,
                    icon=":material/play_arrow:")
    st.markdown("---")

    st.caption("Data    : yfinance  |  " + lookback_label)
    st.caption("Fallbk  : Synthetic MVN (756 days)")
    st.caption("Optim   : SciPy SLSQP  |  long-only")
    st.caption(f"Sim     : {N_SIM:,} Monte Carlo portfolios")
    st.caption("Risk    : 95% Historical VaR & CVaR")
    st.caption("Cov     : Tikhonov regularised (lam=1e-6)")


# ─────────────────────────────────────────────────────────────────────────────
#  5.2  MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>Markowitz Mean-Variance Optimization</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#44446a; font-size:0.82rem; font-weight:600; "
    "margin-top:-12px; letter-spacing:0.08em; text-transform:uppercase;'>"
    "Institutional Portfolio Construction Terminal  &nbsp;|&nbsp;  "
    "SciPy SLSQP  &nbsp;|&nbsp;  95% Historical VaR / CVaR  &nbsp;|&nbsp;  "
    "Developed by Kunal Kaushal"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  5.3  INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
if len(tickers) < 2:
    st.warning(
        "Select at least **2 assets** from the sidebar dropdown "
        "(or add custom tickers) to run the optimisation.",
        icon=":material/warning:",
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  5.4  SESSION-STATE CACHE INVALIDATION
#  Hash (tickers, lookback_label) as run-key. total_capital intentionally
#  excluded — it only affects display, not the optimisation math.
# ─────────────────────────────────────────────────────────────────────────────
_run_key = (tuple(tickers), lookback_label)

if (
    "res"      not in st.session_state
    or run
    or st.session_state.get("_run_key") != _run_key
):
    st.session_state["_run_key"] = _run_key

    with st.spinner(f"Fetching {lookback_label} price data ..."):
        prices, is_synth, _fb_reason = load_price_data(
            tickers, lookback_days, lookback_label
        )

    names = [t.replace(".NS", "").replace(".BO", "") for t in prices.columns]

    with st.spinner("Computing log-returns and annualised statistics ..."):
        lr      = log_returns_df(prices)
        mu, cov = annualize_stats(lr)

    with st.spinner("Running SciPy SLSQP optimization ..."):
        ws = opt_max_sharpe(mu, cov, rf)
        wm = opt_min_var(mu, cov)

    with st.spinner(f"Simulating {N_SIM:,} random portfolios ..."):
        mc_r, mc_v, mc_s = monte_carlo(mu, cov, rf)

    with st.spinner("Rendering Efficient Frontier ..."):
        fig = build_figure(
            mu, cov, ws, wm, mc_r, mc_v, mc_s,
            lr, names, rf, is_synth, lookback_label,
        )

    st.session_state["res"] = dict(
        prices=prices, lr=lr, mu=mu, cov=cov,
        ws=ws, wm=wm, mc_r=mc_r, mc_v=mc_v, mc_s=mc_s,
        names=names, rf=rf, fig=fig, is_synth=is_synth,
        lookback_label=lookback_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5.5  UNPACK SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
R              = st.session_state["res"]
prices         = R["prices"];   lr      = R["lr"]
mu             = R["mu"];       cov     = R["cov"]
ws             = R["ws"];       wm      = R["wm"]
names          = R["names"];    rf      = R["rf"]
fig            = R["fig"];      is_synth= R["is_synth"]
lookback_label = R["lookback_label"]

rs, vs, ss   = p_ret(ws,mu), p_vol(ws,cov), p_sharpe(ws,mu,cov,rf)
rm, vm, sm   = p_ret(wm,mu), p_vol(wm,cov), p_sharpe(wm,mu,cov,rf)
vars_, cvars = hist_var_cvar(ws, lr)
varm,  cvarm = hist_var_cvar(wm, lr)
ar, av       = asset_metrics(mu, cov)


# ─────────────────────────────────────────────────────────────────────────────
#  5.6  THREE-TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_terminal, tab_report = st.tabs([
    "Overview",
    "Portfolio Optimization",
    "Investment Insights",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1  — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    st.markdown("## About This Dashboard")

    st.markdown(
        "<div class='overview-block'>"
        "Developed by <strong>Kunal Kaushal</strong>, a B.Com (Hons) graduate. "
        "This dashboard is a practical application and capstone implementation for the "
        "<em>Advanced Financial Analytics</em> certification via "
        "<strong>IIT Kanpur (NPTEL / Swayam)</strong>. "
        "It demonstrates the deployment of SciPy-based <strong>Sequential Least Squares "
        "Programming (SLSQP)</strong> and <strong>Monte Carlo simulations</strong> for "
        "real-world portfolio optimization."
        "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### Theoretical Foundation")
        st.markdown(
            "<div class='overview-block'>"
            "<strong>Harry Markowitz (1952)</strong> formalized the idea that rational "
            "investors should not simply maximize expected return — they must consider "
            "risk simultaneously. Mean-Variance Optimization (MVO) finds the set of "
            "portfolios that offer the <em>highest return for a given level of risk</em>, "
            "or equivalently, the <em>lowest risk for a given level of return</em>. "
            "This set is called the <strong>Efficient Frontier</strong>.<br><br>"
            "The two portfolios highlighted in this terminal are:<br>"
            "<ul>"
            "<li><strong>Max Sharpe Portfolio</strong> — maximises the Sharpe ratio "
            "(return per unit of risk); the best risk-adjusted portfolio.</li>"
            "<li><strong>Min Variance Portfolio</strong> — sits at the leftmost point "
            "of the frontier; the lowest achievable volatility for a fully-invested "
            "long-only portfolio.</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Mathematical Formulation")
        st.code(
            "Portfolio Return    : R_p  = w^T · mu\n"
            "Portfolio Variance  : s2_p = w^T · Sigma · w\n"
            "Sharpe Ratio        : SR   = (R_p - R_f) / s_p\n"
            "Tikhonov Reg        : Sigma_reg = Sigma + 1e-6 * I\n"
            "SLSQP Constraints   : sum(w_i)=1,  0<=w_i<=1  (long-only)\n"
            "VaR (95%)           : 5th percentile of daily portfolio returns\n"
            "CVaR (95%)          : E[r | r <= VaR]  (expected shortfall)",
            language="text",
        )

    with col_b:
        st.markdown("### Engine Architecture")
        st.markdown(
            "<div class='overview-block'>"
            "<table style='width:100%; border-collapse:collapse; font-size:0.82rem;'>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Optimiser</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>SciPy SLSQP — Sequential "
            "Least Squares Programming (ftol=1e-12, maxiter=2,000)</td></tr>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Simulation</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>Monte Carlo — 5,000 random "
            "Dirichlet-weighted portfolios to map the feasible set</td></tr>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Risk Engine</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>95% Historical VaR & CVaR "
            "(Expected Shortfall) on actual log-return distributions</td></tr>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Covariance</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>Tikhonov (ridge) "
            "regularisation, lambda=1e-6 — guarantees positive-definiteness even "
            "when T~=N (short data windows)</td></tr>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Fallback</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>Synthetic MVN engine — "
            "756 days of simulated multivariate-normal log-returns when yfinance "
            "is offline or lookback is too short</td></tr>"
            "<tr><td style='color:#5555aa; padding:5px 8px; font-weight:700; "
            "text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;'>"
            "Universe</td>"
            "<td style='color:#aaaacc; padding:5px 8px;'>55 NSE tickers — all "
            "Nifty 50 constituents + select high-momentum Indian large-caps across "
            "12 sectors; .NS auto-appended</td></tr>"
            "</table>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Key Capabilities")
        capabilities = [
            ("Dynamic Lookback",    "1 Day to 5 Years — configurable from the sidebar"),
            ("Offline Resilience",  "Synthetic MVN fallback preserves full math fidelity"),
            ("Capital Allocation",  "Rupee-denominated allocation for any investment size"),
            ("Plain-English Report","Layman-friendly interpretation of all results"),
            ("PyArrow Safety",      "All DataFrames use explicit float64 — Arrow compatible"),
            ("1-Hour Cache",        "yfinance results cached via st.cache_data(ttl=3600)"),
        ]
        for title, desc in capabilities:
            st.markdown(
                f"<div style='display:flex; align-items:flex-start; margin-bottom:8px;'>"
                f"<span style='color:#3a3a88; font-size:0.9rem; margin-right:10px; "
                f"margin-top:1px; flex-shrink:0;'>▸</span>"
                f"<span style='font-size:0.83rem;'>"
                f"<strong style='color:#8888cc;'>{title}</strong>"
                f"<span style='color:#555577;'> — </span>"
                f"<span style='color:#888899;'>{desc}</span></span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#22223a; font-size:0.72rem; "
        "letter-spacing:0.06em; font-family:monospace;'>"
        "MARKOWITZ MVO DASHBOARD  |  SCIPY SLSQP  |  95% HISTORICAL VAR/CVAR  |  "
        "TIKHONOV COVARIANCE (LAM=1E-6)  |  YFINANCE  |  "
        "FOR INFORMATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2  — QUANT TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_terminal:

    # Data source banner
    if is_synth:
        st.info(
            f"**Simulated Data Mode** — the '{lookback_label}' window returned "
            "insufficient live data (too few rows or yfinance offline). "
            "All metrics use **756 days of synthetic multivariate-normal "
            "log-returns**. Optimisation math is identical; only the price "
            "series is artificial.",
            icon=":material/science:",
        )
    else:
        st.success(
            f"**Live Data** — {len(prices):,} clean trading days loaded from "
            f"yfinance  |  Lookback: **{lookback_label}**",
            icon=":material/check_circle:",
        )

    # Metadata strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data Start",   prices.index[0].strftime("%d %b %Y"))
    c2.metric("Data End",     prices.index[-1].strftime("%d %b %Y"))
    c3.metric("Trading Days", f"{len(prices):,}")
    c4.metric("Assets",       str(len(names)))
    c5.metric("Lookback",     lookback_label)
    st.markdown("---")

    # ── Efficient Frontier Chart ─────────────────────────────────────────────
    st.markdown("## Efficient Frontier")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # ── Capital Allocation Summary ───────────────────────────────────────────
    st.markdown("## Capital Allocation Summary")
    st.caption(
        f"Based on total investment capital of **Rs. {total_capital:,.0f}**. "
        "Stocks with allocation < 0.01% are filtered out for clarity. "
        "Adjust capital using the sidebar slider."
    )

    col_sharpe, col_minvar = st.columns(2, gap="large")

    # Helper: build allocation dataframe for one portfolio
    def _build_alloc_df(weights: np.ndarray, asset_names: list[str],
                        capital: float, min_wt: float = 0.0001) -> pd.DataFrame:
        mask = weights >= min_wt
        tickers_f = [asset_names[i] for i in range(len(asset_names)) if mask[i]]
        weights_f = weights[mask]
        rupees_f  = weights_f * capital
        return pd.DataFrame(
            {
                "Ticker"            : tickers_f,
                "Weight (%)"        : (weights_f * 100).astype(float),
                "Allocation (Rs.)"  : rupees_f.astype(float),
            }
        ).set_index("Ticker")

    with col_sharpe:
        st.markdown(
            "<div class='alloc-header-sharpe'>Best Balance — Max Sharpe Portfolio</div>",
            unsafe_allow_html=True,
        )
        ms1, ms2 = st.columns(2)
        ms1.metric("Expected Return",   f"{rs*100:.2f}%")
        ms2.metric("Risk / Volatility", f"{vs*100:.2f}%")
        ms1.metric("Sharpe Ratio",      f"{ss:.3f}")
        ms2.metric("95% Daily VaR",     f"{vars_*100:.3f}%")

        sharpe_alloc = _build_alloc_df(ws, names, total_capital)
        st.dataframe(
            sharpe_alloc,
            use_container_width=True,
            height=min(400, (len(sharpe_alloc) + 2) * 38),
            column_config={
                "Weight (%)"       : st.column_config.NumberColumn(
                                        "Weight (%)", format="%.2f"),
                "Allocation (Rs.)" : st.column_config.NumberColumn(
                                        "Allocation (Rs.)", format="Rs. %,.0f"),
            },
        )
        st.caption(
            f"Sum of weights: **{ws.sum()*100:.4f}%**  |  "
            f"Total allocated: **Rs. {total_capital:,.0f}**"
        )

    with col_minvar:
        st.markdown(
            "<div class='alloc-header-minvar'>Safest Portfolio — Min Variance</div>",
            unsafe_allow_html=True,
        )
        mv1, mv2 = st.columns(2)
        mv1.metric("Expected Return",   f"{rm*100:.2f}%")
        mv2.metric("Risk / Volatility", f"{vm*100:.2f}%")
        mv1.metric("Sharpe Ratio",      f"{sm:.3f}")
        mv2.metric("95% Daily VaR",     f"{varm*100:.3f}%")

        minvar_alloc = _build_alloc_df(wm, names, total_capital)
        st.dataframe(
            minvar_alloc,
            use_container_width=True,
            height=min(400, (len(minvar_alloc) + 2) * 38),
            column_config={
                "Weight (%)"       : st.column_config.NumberColumn(
                                        "Weight (%)", format="%.2f"),
                "Allocation (Rs.)" : st.column_config.NumberColumn(
                                        "Allocation (Rs.)", format="Rs. %,.0f"),
            },
        )
        st.caption(
            f"Sum of weights: **{wm.sum()*100:.4f}%**  |  "
            f"Total allocated: **Rs. {total_capital:,.0f}**"
        )

    st.markdown("---")

    # ── Performance Summary Grid ─────────────────────────────────────────────
    st.markdown("## Performance Summary")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Max Sharpe Portfolio")
        a, b = st.columns(2)
        a.metric("Annual Return",     f"{rs*100:.4f}%",
                 f"{(rs-rm)*100:+.4f}% vs MinVar")
        b.metric("Annual Volatility", f"{vs*100:.4f}%",
                 f"{(vs-vm)*100:+.4f}% vs MinVar")
        a.metric("Sharpe Ratio",      f"{ss:.4f}",
                 f"{ss-sm:+.4f} vs MinVar")
        b.metric("Risk-Free Rate",    f"{rf*100:.2f}%")
        a.metric("95% Daily VaR",     f"{vars_*100:.4f}%",
                 f"{(vars_-varm)*100:+.4f}% vs MinVar")
        b.metric("95% Daily CVaR",    f"{cvars*100:.4f}%",
                 f"{(cvars-cvarm)*100:+.4f}% vs MinVar")

    with right:
        st.markdown("### Min Variance Portfolio")
        c, d = st.columns(2)
        c.metric("Annual Return",     f"{rm*100:.4f}%",
                 f"{(rm-rs)*100:+.4f}% vs MaxSR")
        d.metric("Annual Volatility", f"{vm*100:.4f}%",
                 f"{(vm-vs)*100:+.4f}% vs MaxSR")
        c.metric("Sharpe Ratio",      f"{sm:.4f}",
                 f"{sm-ss:+.4f} vs MaxSR")
        d.metric("Risk-Free Rate",    f"{rf*100:.2f}%")
        c.metric("95% Daily VaR",     f"{varm*100:.4f}%",
                 f"{(varm-vars_)*100:+.4f}% vs MaxSR")
        d.metric("95% Daily CVaR",    f"{cvarm*100:.4f}%",
                 f"{(cvarm-cvars)*100:+.4f}% vs MaxSR")

    st.markdown("---")

    # ── Core Metrics Breakdown ───────────────────────────────────────────────
    st.markdown("## Core Metrics Breakdown")

    metrics_df = pd.DataFrame(
        {
            "Max Sharpe"  : [rs*100, vs*100, ss,    rf*100, vars_*100, cvars*100],
            "Min Variance": [rm*100, vm*100, sm,    rf*100, varm*100,  cvarm*100],
        },
        index=pd.Index(
            ["Annual Return (%)", "Annual Volatility (%)", "Sharpe Ratio",
             "Risk-Free Rate (%)", "95% Daily VaR (%)", "95% Daily CVaR (%)"],
            name="Metric",
        ),
        dtype=float,
    )
    st.dataframe(
        metrics_df,
        use_container_width=True,
        height=260,
        column_config={
            "Max Sharpe"  : st.column_config.NumberColumn("Max Sharpe",   format="%.4f"),
            "Min Variance": st.column_config.NumberColumn("Min Variance", format="%.4f"),
        },
    )
    st.markdown("---")

    # ── Full Asset Allocation Table ──────────────────────────────────────────
    st.markdown("## Asset Allocation Detail")

    alloc_df = pd.DataFrame(
        {
            "Max Sharpe Wt (%)"  : (ws * 100).astype(float),
            "Min Variance Wt (%)": (wm * 100).astype(float),
            "Indiv. Return (%)"  : (ar * 100).astype(float),
            "Indiv. Vol (%)"     : (av * 100).astype(float),
        },
        index=pd.Index(names, name="Ticker"),
        dtype=float,
    )
    st.dataframe(
        alloc_df,
        use_container_width=True,
        height=(len(names) + 2) * 38,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4f")
            for col in alloc_df.columns
        },
    )
    tc1, tc2 = st.columns(2)
    tc1.metric("Sum of Max Sharpe Weights",   f"{ws.sum()*100:.6f}%")
    tc2.metric("Sum of Min Variance Weights", f"{wm.sum()*100:.6f}%")
    st.markdown("---")

    # ── Individual Asset Risk Metrics ────────────────────────────────────────
    st.markdown("## Individual Asset Risk Metrics")

    _var_l, _cvar_l = [], []
    for i in range(len(names)):
        w_s = np.zeros(len(names)); w_s[i] = 1.0
        v, cv = hist_var_cvar(w_s, lr)
        _var_l.append(float(v * 100))
        _cvar_l.append(float(cv * 100))

    asset_risk_df = pd.DataFrame(
        {
            "Annual Return (%)"  : (ar * 100).astype(float),
            "Annual Vol (%)"     : (av * 100).astype(float),
            "95% Daily VaR (%)"  : _var_l,
            "95% Daily CVaR (%)" : _cvar_l,
        },
        index=pd.Index(names, name="Ticker"),
        dtype=float,
    )
    st.dataframe(
        asset_risk_df,
        use_container_width=True,
        height=(len(names) + 2) * 38,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4f")
            for col in asset_risk_df.columns
        },
    )
    st.markdown("---")

    # Footer
    st.markdown(
        "<p style='text-align:center; color:#1a1a32; font-size:0.70rem; "
        "font-family:monospace; letter-spacing:0.05em;'>"
        f"MARKOWITZ MVO  |  SCIPY SLSQP  |  95% HISTORICAL VAR/CVAR  |  "
        f"LOOKBACK: {lookback_label.upper()}  |  NIFTY 50 + HIGH-MOMENTUM (55 TICKERS)  |  "
        "AUTO .NS REPAIR  |  SYNTHETIC MVN FALLBACK (756 DAYS)  |  "
        "TIKHONOV REGULARISATION (LAM=1E-6)  |  DATA: YFINANCE  |  "
        "FOR INFORMATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3  — PLAIN-ENGLISH PORTFOLIO REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:

    st.markdown("## Plain-English Portfolio Report")
    st.caption(
        "This report translates the quantitative results from the Quant Terminal "
        "into everyday language — the way a financial advisor would explain "
        "your portfolio to you in a meeting."
    )

    if "res" not in st.session_state:
        st.info(
            "Run the optimization from the Quant Terminal tab first to generate "
            "this report.",
            icon=":material/info:",
        )
        st.stop()

    # ── Identify top 2-3 holdings per portfolio ───────────────────────────────
    sharpe_sorted  = sorted(
        zip(names, ws), key=lambda x: x[1], reverse=True
    )
    minvar_sorted  = sorted(
        zip(names, wm), key=lambda x: x[1], reverse=True
    )
    top_sharpe     = [(n, w) for n, w in sharpe_sorted if w > 0.0001][:3]
    top_minvar     = [(n, w) for n, w in minvar_sorted if w > 0.0001][:3]

    # Dominant stock for Max Sharpe
    dominant_sharpe = top_sharpe[0][0] if top_sharpe else names[0]
    dominant_pct_s  = top_sharpe[0][1] * 100 if top_sharpe else 0.0
    top2_names_s    = ", ".join(n for n, _ in top_sharpe[:2])

    # Dominant stock for Min Variance
    dominant_minvar = top_minvar[0][0] if top_minvar else names[0]
    dominant_pct_m  = top_minvar[0][1] * 100 if top_minvar else 0.0
    top2_names_m    = ", ".join(n for n, _ in top_minvar[:2])

    # Capital-based values
    cap_rs_inr   = total_capital * rs          # expected gain (Max Sharpe)
    cap_rm_inr   = total_capital * rm          # expected gain (Min Variance)
    cap_var_s    = abs(vars_) * total_capital  # VaR rupees (Max Sharpe)
    cap_var_m    = abs(varm) * total_capital   # VaR rupees (Min Variance)
    cap_cvar_s   = abs(cvars) * total_capital  # CVaR rupees (Max Sharpe)

    data_context = (
        f"{'Simulated (Synthetic MVN)' if is_synth else 'Live NSE Data'}  "
        f"|  Lookback: {lookback_label}  |  {len(names)} assets  |  "
        f"Rs. {total_capital:,.0f} capital"
    )
    st.caption(f"**Data context:** {data_context}")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    #  Section A — Max Sharpe Portfolio
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### A.   What Your Best Balance Portfolio Actually Means")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Max Sharpe Portfolio — Best Risk-Adjusted Balance</h4>",
            unsafe_allow_html=True,
        )

        # Metric strip
        ra1, ra2, ra3 = st.columns(3)
        ra1.metric("Expected Annual Return", f"{rs*100:.2f}%")
        ra2.metric("Annual Risk (Volatility)", f"{vs*100:.2f}%")
        ra3.metric("Sharpe Ratio", f"{ss:.3f}")

        if rs > 0:
            narrative_a = (
                f"If you invest **Rs. {total_capital:,.0f}** into this portfolio, "
                f"the mathematical model — built from {lookback_label.lower()} of "
                f"{'simulated' if is_synth else 'real'} market data — projects an "
                f"expected annual gain of approximately **Rs. {cap_rs_inr:,.0f}** "
                f"({rs*100:.1f}% return). In practical terms, that means your "
                f"Rs. {total_capital:,.0f} could grow to roughly "
                f"**Rs. {total_capital + cap_rs_inr:,.0f}** after one year, "
                f"assuming conditions stay close to the historical average.\n\n"
                f"However, this is not a guaranteed outcome. The portfolio carries a "
                f"risk (annual volatility) of **{vs*100:.1f}%**. Think of volatility "
                f"as the normal range of ups and downs your investment will experience "
                f"throughout the year. On a good year you could exceed the expected "
                f"return; on a poor year your portfolio could decline. The **Sharpe "
                f"Ratio of {ss:.2f}** tells you that for every unit of risk you are "
                f"taking on, you are being compensated with {ss:.2f} units of return "
                f"above the risk-free rate ({rf*100:.2f}%). A Sharpe above 1.0 is "
                f"generally considered good; above 2.0 is excellent."
            )
        else:
            narrative_a = (
                f"This portfolio has a negative expected return of **{rs*100:.2f}%** "
                f"for the selected lookback window. This may reflect a difficult market "
                f"period in the data. The Sharpe Ratio of {ss:.2f} should be interpreted "
                f"with caution. Consider extending the lookback period or changing the "
                f"asset selection to include stocks with better historical performance."
            )

        st.markdown(narrative_a)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Section B — Min Variance Portfolio
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### B.   What the Safest Portfolio Means, and Why It Is Safer")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Min Variance Portfolio — Lowest Achievable Risk</h4>",
            unsafe_allow_html=True,
        )

        rb1, rb2, rb3 = st.columns(3)
        rb1.metric("Expected Annual Return", f"{rm*100:.2f}%")
        rb2.metric("Annual Risk (Volatility)", f"{vm*100:.2f}%")
        rb3.metric("Sharpe Ratio", f"{sm:.3f}")

        vol_diff  = (vs - vm) * 100
        ret_diff  = (rs - rm) * 100

        narrative_b = (
            f"The Safest Portfolio (Minimum Variance) is designed for one purpose: "
            f"**minimise how much your portfolio swings in value**, regardless of return. "
            f"It achieves an annual volatility of **{vm*100:.1f}%** — "
            f"that is **{abs(vol_diff):.1f} percentage points {'lower' if vol_diff > 0 else 'higher'} "
            f"than the Max Sharpe portfolio** ({vs*100:.1f}%).\n\n"
            f"The trade-off is return: this portfolio targets **{rm*100:.1f}% annual return**, "
            f"versus {rs*100:.1f}% for the Max Sharpe. For your Rs. {total_capital:,.0f} "
            f"investment, the expected gain is approximately **Rs. {cap_rm_inr:,.0f}** "
            f"— Rs. {abs(cap_rs_inr - cap_rm_inr):,.0f} "
            f"{'less' if cap_rs_inr > cap_rm_inr else 'more'} than the Max Sharpe portfolio.\n\n"
            f"**Why choose this?** If you are a conservative investor, nearing retirement, "
            f"or simply cannot stomach large daily fluctuations in your portfolio value, "
            f"the Min Variance portfolio will let you sleep better at night. You sacrifice "
            f"some upside potential, but you significantly reduce the risk of a painful "
            f"drawdown in a volatile market."
        )
        st.markdown(narrative_b)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Section C — VaR in Rupee Terms
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### C.   Your Value at Risk in Actual Rupees")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>95% Historical Value at Risk — Daily Loss Threshold</h4>",
            unsafe_allow_html=True,
        )

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Max Sharpe VaR (%)",   f"{vars_*100:.3f}%")
        rc2.metric("Max Sharpe VaR (Rs.)", f"Rs. {cap_var_s:,.0f}")
        rc3.metric("Min Variance VaR (%)", f"{varm*100:.3f}%")
        rc4.metric("Min Variance VaR (Rs.)",f"Rs. {cap_var_m:,.0f}")

        narrative_c = (
            f"**Value at Risk (VaR)** answers a simple question: "
            f"*How much money could I lose on a really bad day?*\n\n"
            f"Based on your **Rs. {total_capital:,.0f}** investment in the "
            f"**Max Sharpe portfolio**, the 95% daily VaR is approximately "
            f"**Rs. {cap_var_s:,.0f}** ({abs(vars_*100):.3f}% of capital). "
            f"This means that on **95 out of every 100 trading days**, your "
            f"single-day loss should NOT exceed Rs. {cap_var_s:,.0f}. "
            f"The remaining 5% of days — roughly 12-13 trading days per year — "
            f"could see losses larger than this threshold.\n\n"
            f"For the **Min Variance portfolio**, that same protection applies at "
            f"**Rs. {cap_var_m:,.0f} per day** ({abs(varm*100):.3f}%), which is "
            f"Rs. {abs(cap_var_s - cap_var_m):,.0f} "
            f"{'less' if cap_var_m < cap_var_s else 'more'} per day — confirming "
            f"that the Safest Portfolio is genuinely safer on a daily loss basis too.\n\n"
            f"The CVaR (Conditional VaR, or Expected Shortfall) for Max Sharpe is "
            f"**Rs. {cap_cvar_s:,.0f}**, which is the *average* loss you would "
            f"expect on those rare worst-case days when the VaR threshold is breached. "
            f"This is the number stress-testing frameworks use to size capital buffers."
        )
        st.markdown(narrative_c)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Section D — Why the Optimizer Concentrated in Top Holdings
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### D.   Why Certain Stocks Dominate Your Final Allocation")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Understanding Concentration — The SLSQP Optimizer's Logic</h4>",
            unsafe_allow_html=True,
        )

        rd1, rd2 = st.columns(2)
        with rd1:
            st.markdown("**Max Sharpe — Top Holdings**")
            for i, (nm, wt) in enumerate(top_sharpe, 1):
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:4px 0; border-bottom:1px solid #14143a;'>"
                    f"<span style='color:#8888cc; font-size:0.82rem; font-family:monospace;'>"
                    f"#{i}  {nm}</span>"
                    f"<span style='color:#EF5350; font-size:0.82rem; font-weight:700;'>"
                    f"{wt*100:.1f}%  —  Rs. {wt*total_capital:,.0f}</span></div>",
                    unsafe_allow_html=True,
                )
        with rd2:
            st.markdown("**Min Variance — Top Holdings**")
            for i, (nm, wt) in enumerate(top_minvar, 1):
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:4px 0; border-bottom:1px solid #14143a;'>"
                    f"<span style='color:#8888cc; font-size:0.82rem; font-family:monospace;'>"
                    f"#{i}  {nm}</span>"
                    f"<span style='color:#26A69A; font-size:0.82rem; font-weight:700;'>"
                    f"{wt*100:.1f}%  —  Rs. {wt*total_capital:,.0f}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")  # spacer

        narrative_d = (
            f"You might look at your portfolio and wonder: *why is so much money "
            f"concentrated in just {len(top_sharpe)} or {len(top_minvar)} stocks?*\n\n"
            f"The answer lies in the mathematics of the SLSQP optimizer. It is "
            f"solving a precise mathematical problem: *given the historical returns "
            f"and correlations of all {len(names)} selected stocks, find the exact "
            f"combination of weights that maximises the Sharpe ratio (or minimises "
            f"variance).* The optimizer does not care about equal distribution — "
            f"it cares about mathematical efficiency.\n\n"
            f"**{top2_names_s}** emerged as dominant holdings in the Max Sharpe "
            f"portfolio because, historically over the selected "
            f"{lookback_label.lower()} period, they offered the best combination "
            f"of high return AND low correlation with each other. When two assets "
            f"move in different directions (low or negative correlation), holding "
            f"both simultaneously *reduces total portfolio risk without sacrificing "
            f"proportionate return* — this is the core of Markowitz diversification.\n\n"
            f"Stocks with lower individual Sharpe ratios, or those highly correlated "
            f"with the dominant holdings, received near-zero weights because they "
            f"would have added risk without adding proportionate return. Think of it "
            f"this way: if two stocks move almost identically, there is no "
            f"diversification benefit to owning both — the optimizer simply picks "
            f"the better one.\n\n"
            f"**Important caveat:** These results are based on *historical* data "
            f"from the {lookback_label.lower()} window. Past correlations and returns "
            f"do not guarantee future performance. Longer lookback periods generally "
            f"produce more stable and reliable weight estimates."
        )
        st.markdown(narrative_d)
        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#1e1e38; font-size:0.70rem; "
        "font-family:monospace; letter-spacing:0.05em;'>"
        "THIS REPORT IS FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY. "
        "IT DOES NOT CONSTITUTE FINANCIAL ADVICE, AN INVESTMENT RECOMMENDATION, "
        "OR A SOLICITATION TO BUY OR SELL ANY SECURITY. "
        "PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE RESULTS. "
        "CONSULT A QUALIFIED FINANCIAL ADVISOR BEFORE MAKING INVESTMENT DECISIONS."
        "</p>",
        unsafe_allow_html=True,
    )
