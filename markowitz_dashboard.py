"""
================================================================================
  MARKOWITZ MEAN-VARIANCE OPTIMIZATION  |  PORTFOLIO ANALYTICS TERMINAL
  ───────────────────────────────────────────────────────────────────────
  Developed  : Kunal Kaushal
  Programme  : Advanced Financial Analytics — IIT Kanpur (NPTEL / Swayam)
  Version    : v3.0  |  IIT Kanpur NPTEL Capstone
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
  §5  Streamlit UI     — 3-tab layout: Overview | Portfolio Optimization | Insights

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
    page_title            = "Portfolio Analytics | Kunal Kaushal",
    page_icon             = "◈",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  §A  NIFTY SECTOR POOL  — 55-ticker NSE universe
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_SECTOR_POOL: list[str] = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",     "AXISBANK.NS",  "KOTAKBANK.NS",
    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "IRFC.NS",     "JIOFIN.NS",
    "TCS.NS",      "INFY.NS",     "HCLTECH.NS",   "WIPRO.NS",     "TECHM.NS",
    "RELIANCE.NS", "ONGC.NS",     "BPCL.NS",      "NTPC.NS",      "POWERGRID.NS",
    "COALINDIA.NS","SUZLON.NS",
    "TATAMOTORS.NS","MARUTI.NS",  "M&M.NS",       "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "LT.NS",       "ADANIENT.NS", "ADANIPORTS.NS","BEL.NS",       "HAL.NS",
    "SUNPHARMA.NS","DRREDDY.NS",  "CIPLA.NS",     "APOLLOHOSP.NS","DIVISLAB.NS",
    "HINDUNILVR.NS","ITC.NS",     "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "TITAN.NS",    "ASIANPAINT.NS","ZOMATO.NS",   "TRENT.NS",
    "BHARTIARTL.NS",
    "ULTRACEMCO.NS","GRASIM.NS",
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
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── canvas ───────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background:#08080f; }
[data-testid="stSidebar"]          { background:#0a0a16;
                                     border-right:1px solid #14143a; }
[data-testid="stHeader"]           { background:#08080f; }

/* ── disclaimer banner ────────────────────────────────────────────────── */
.disclaimer-banner {
    background    : #0e0e1a;
    border        : 1px solid #1e1e3a;
    border-left   : 3px solid #3a3a6a;
    border-radius : 0 6px 6px 0;
    padding       : 8px 16px;
    font-size     : 0.72rem;
    color         : #55557a !important;
    letter-spacing: 0.04em;
    margin-bottom : 14px;
}

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
    color         : #44446a !important;
    font-size     : 0.80rem !important;
    font-weight   : 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding       : 10px 24px !important;
    transition    : all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color         : #8080bb !important;
    border-bottom : 2px solid #2a2a70 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color         : #a0a8ff !important;
    border-bottom : 2px solid #4444aa !important;
    background    : #0c0c1e !important;
}

/* ── metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background    : #0d0d1c;
    border        : 1px solid #1c1c3c;
    border-radius : 10px;
    padding       : 14px 18px;
    box-shadow    : 0 2px 10px rgba(15,15,60,0.3);
    transition    : all 0.3s cubic-bezier(0.4,0,0.2,1);
    position      : relative;
    overflow      : hidden;
}
[data-testid="metric-container"]::before {
    content    : '';
    position   : absolute;
    top:0; left:0; right:0; height:2px;
    background : linear-gradient(90deg, #222266, #4444aa, #4488cc);
    opacity    : 0;
    transition : opacity 0.3s ease;
}
[data-testid="metric-container"]:hover {
    border-color : #2a2a60;
    box-shadow   : 0 4px 20px rgba(40,40,120,0.35);
    transform    : translateY(-1px);
}
[data-testid="metric-container"]:hover::before { opacity:1; }
[data-testid="metric-container"] label {
    color         : #4a4a7a !important;
    font-size     : 0.70rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color        : #d8dcf8;
    font-size    : 1.15rem;
    font-weight  : 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.73rem;
}

/* ── inputs ───────────────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div {
    background  : #0d0d1c !important;
    border      : 1px solid #1e1e4a !important;
    border-radius: 8px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background  : #181855 !important;
    border      : 1px solid #2a2a70 !important;
    border-radius: 4px !important;
    color       : #9090cc !important;
    font-size   : 0.70rem !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg { fill:#5555aa !important; }
[data-testid="stSelectbox"] > div > div {
    background  : #0d0d1c !important;
    border      : 1px solid #1e1e4a !important;
    border-radius: 8px !important;
    color       : #9090cc !important;
}
[data-testid="stTextInput"] input {
    background  : #0d0d1c !important;
    border      : 1px solid #1e1e4a !important;
    border-radius: 8px !important;
    color       : #c0c4ff !important;
}
[data-testid="stExpander"] {
    background  : #0c0c1a !important;
    border      : 1px solid #16163a !important;
    border-radius: 8px !important;
}
[data-testid="stDataFrame"] {
    border      : 1px solid #1a1a3a;
    border-radius: 8px;
}

/* ── typography ───────────────────────────────────────────────────────── */
h1 { color:#a0a8ff !important; font-size:1.65rem !important; letter-spacing:-0.01em !important; }
h2 { color:#6868aa !important; font-size:1.0rem !important; letter-spacing:0.08em !important;
     text-transform:uppercase !important; }
h3 { color:#5050888 !important; font-size:0.96rem !important; }
p, li, label, span, div { color:#a8aac8 !important; }
hr { border-color:#10103a; }
code { color:#6699ee !important; background:#0c0c22 !important; }
strong { color:#c0c4f0 !important; }

/* ── run button ───────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background  : linear-gradient(135deg,#1e1e66,#2e2e88);
    color       : #c0c8ff;
    border      : 1px solid #2e2e70;
    border-radius: 8px;
    font-weight : 600;
    padding     : 10px 0;
    width       : 100%;
    letter-spacing: 0.08em;
    font-size   : 0.80rem;
    transition  : all .25s;
}
[data-testid="stButton"] > button:hover {
    background  : linear-gradient(135deg,#262688,#3838aa);
    border-color: #4444aa;
    box-shadow  : 0 0 16px rgba(50,50,160,0.4);
    color       : #ffffff;
}

/* ── sector pills ─────────────────────────────────────────────────────── */
.sector-pill {
    display      : inline-block;
    background   : #10103a;
    border       : 1px solid #22225a;
    border-radius: 3px;
    padding      : 1px 6px;
    font-size    : 0.65rem;
    color        : #6666aa !important;
    margin-right : 3px;
    margin-bottom: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── lookback warning ─────────────────────────────────────────────────── */
.lookback-warn {
    background  : #14100a;
    border      : 1px solid #3a2a00;
    border-radius: 6px;
    padding     : 6px 10px;
    font-size   : 0.72rem;
    color       : #998833 !important;
    margin-top  : 4px;
}

/* ── chart caption ────────────────────────────────────────────────────── */
.chart-caption {
    background    : #0c0c1e;
    border        : 1px solid #18184a;
    border-left   : 3px solid #3333aa;
    border-radius : 0 6px 6px 0;
    padding       : 10px 16px;
    font-size     : 0.82rem;
    color         : #7070aa !important;
    margin-bottom : 8px;
    line-height   : 1.6;
}

/* ── decision box ─────────────────────────────────────────────────────── */
.decision-box {
    background    : #0c0c20;
    border        : 1px solid #20205a;
    border-radius : 10px;
    padding       : 18px 24px;
    margin-bottom : 20px;
}
.decision-box-title {
    font-size     : 0.65rem;
    font-weight   : 700;
    letter-spacing: 0.14em;
    color         : #4444aa !important;
    text-transform: uppercase;
    margin-bottom : 14px;
}
.decision-pill-growth {
    display       : inline-block;
    background    : #1a1030;
    border        : 1px solid #4a2a70;
    border-radius : 6px;
    padding       : 10px 16px;
    margin-right  : 10px;
    margin-bottom : 8px;
    font-size     : 0.80rem;
}
.decision-pill-safety {
    display       : inline-block;
    background    : #0a1e1c;
    border        : 1px solid #1a5050;
    border-radius : 6px;
    padding       : 10px 16px;
    margin-right  : 10px;
    margin-bottom : 8px;
    font-size     : 0.80rem;
}

/* ── allocation headers ───────────────────────────────────────────────── */
.alloc-header-sharpe {
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.12em;
    color         : #cc4444 !important;
    text-transform: uppercase;
    border-bottom : 1px solid #2a1414;
    padding-bottom: 6px;
    margin-bottom : 12px;
}
.alloc-header-minvar {
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.12em;
    color         : #2a8a80 !important;
    text-transform: uppercase;
    border-bottom : 1px solid #142222;
    padding-bottom: 6px;
    margin-bottom : 12px;
}

/* ── overview block ───────────────────────────────────────────────────── */
.overview-block {
    background    : #0c0c1e;
    border        : 1px solid #1a1a40;
    border-left   : 3px solid #2a2a88;
    border-radius : 0 8px 8px 0;
    padding       : 18px 24px;
    margin-bottom : 16px;
    line-height   : 1.8;
}

/* ── report card ──────────────────────────────────────────────────────── */
.report-card {
    background    : #0c0c1e;
    border        : 1px solid #1a1a40;
    border-radius : 10px;
    padding       : 20px 26px;
    margin-bottom : 18px;
}
.report-card h4 {
    color         : #5555aa !important;
    font-size     : 0.65rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom : 10px;
    border-bottom : 1px solid #16163a;
    padding-bottom: 6px;
}

/* ── pull quote ───────────────────────────────────────────────────────── */
.pull-quote {
    font-size     : 1.4rem;
    font-weight   : 700;
    color         : #a0a8ff !important;
    letter-spacing: -0.02em;
    margin        : 10px 0 4px 0;
    line-height   : 1.2;
}
.pull-quote-label {
    font-size     : 0.65rem;
    color         : #44447a !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom : 12px;
}

/* ── progress bar risk label ──────────────────────────────────────────── */
.risk-bar-wrap {
    background    : #10102a;
    border-radius : 4px;
    height        : 6px;
    width         : 100%;
    margin        : 6px 0 2px 0;
    overflow      : hidden;
}
.risk-bar-fill-low  { height:6px; border-radius:4px; background:#26A69A; }
.risk-bar-fill-mid  { height:6px; border-radius:4px; background:#FFB300; }
.risk-bar-fill-high { height:6px; border-radius:4px; background:#EF5350; }

/* ── identity wordmark ────────────────────────────────────────────────── */
.wordmark {
    font-size     : 0.62rem;
    letter-spacing: 0.18em;
    color         : #28285a !important;
    text-transform: uppercase;
    font-family   : monospace;
    margin-top    : -10px;
    margin-bottom : 18px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  §1  TICKER REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def _repair_tickers(raw: str) -> list[str]:
    """Parse free-text ticker input; bare Indian names get .NS appended."""
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
#  §2  DATA LAYER
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
    """Synthetic price DataFrame via multivariate-normal log-return simulation."""
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
    """Live yfinance download with synthetic MVN fallback.
    CRITICAL dropna guard prevents matmul dimension errors."""
    n_assets = len(tickers)
    min_rows = max(n_assets + 2, _MIN_ROWS_FLOOR)

    try:
        raw = _yf_download(tuple(tickers), lookback_days)
        # CRITICAL: drop all-NaN columns first, then drop any remaining NaN rows
        raw = raw.dropna(axis=1, how="all").dropna()

        if raw.shape[1] < 2:
            raise ValueError(f"Fewer than 2 tickers returned data for '{lookback_label}'.")
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
    return np.log(prices / prices.shift(1)).dropna()

def regularize_cov(cov: np.ndarray, lam: float = _COV_REGULARISER) -> np.ndarray:
    return cov + lam * np.eye(cov.shape[0])

def annualize_stats(lr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu  = lr.mean().values * TRADING_DAYS
    cov = regularize_cov(lr.cov().values * TRADING_DAYS)
    return mu, cov

def p_ret(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)

def p_vol(w: np.ndarray, cov: np.ndarray) -> float:
    v = float(w @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))

def p_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = p_vol(w, cov)
    return (p_ret(w, mu) - rf) / vol if vol > 1e-12 else 0.0

def _cb(n: int):
    return (
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        tuple((0.0, 1.0) for _ in range(n)),
    )

def opt_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
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
        w = np.clip(res.x, 0, 1); w /= w.sum()
        return w
    except Exception:
        return w0

def opt_min_var(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
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
        w = np.clip(res.x, 0, 1); w /= w.sum()
        return w
    except Exception:
        return w0

def monte_carlo(
    mu: np.ndarray, cov: np.ndarray, rf: float,
    n: int = N_SIM, seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N   = len(mu)
    R, V, S = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        w = rng.random(N); w /= w.sum()
        R[i] = p_ret(w, mu); V[i] = p_vol(w, cov); S[i] = p_sharpe(w, mu, cov, rf)
    return R, V, S

def asset_metrics(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(mu)
    rets, vols = np.empty(n), np.empty(n)
    for i in range(n):
        w = np.zeros(n); w[i] = 1.0
        rets[i] = p_ret(w, mu); vols[i] = p_vol(w, cov)
    return rets, vols

def hist_var_cvar(
    w: np.ndarray, lr: pd.DataFrame, conf: float = 0.95,
) -> tuple[float, float]:
    pret  = lr.values @ w
    alpha = 1.0 - conf
    var   = float(np.percentile(pret, alpha * 100))
    tail  = pret[pret <= var]
    cvar  = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


# ══════════════════════════════════════════════════════════════════════════════
#  §4  PLOTTING LAYER — Plotly interactive efficient frontier
# ══════════════════════════════════════════════════════════════════════════════

_PALETTE = [
    "#4FC3F7","#FFD54F","#F48FB1","#FFAB40","#69F0AE",
    "#CE93D8","#4DD0E1","#FF7043","#D4E157","#80DEEA",
    "#FF8A65","#9575CD","#4DB6AC","#FFF176","#F06292",
    "#A5D6A7","#FFB74D","#B3E5FC","#FF8A80","#82B1FF",
    "#CCFF90","#EA80FC","#84FFFF","#FF6E40","#40C4FF",
    "#B9F6CA","#FF4081","#EEFF41","#7C4DFF","#64FFDA",
    "#FFD180","#CCCCFF","#AEEA00","#FF6D00","#00B0FF",
]


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
    """Interactive Plotly Efficient Frontier — dark terminal aesthetic.
    Hover tooltips on every element. Crosshairs at optimal portfolio points.
    plot_bgcolor / paper_bgcolor = #08080f. Gridlines at #101025."""

    rs, vs, ss = p_ret(ws, mu), p_vol(ws, cov), p_sharpe(ws, mu, cov, rf)
    rm, vm, sm = p_ret(wm, mu), p_vol(wm, cov), p_sharpe(wm, mu, cov, rf)
    ar, av     = asset_metrics(mu, cov)

    all_v = np.concatenate([mc_v*100, av*100, [vs*100, vm*100]])
    all_r = np.concatenate([mc_r*100, ar*100, [rs*100, rm*100]])
    vlo, vhi = float(all_v.min()), float(all_v.max())
    rlo, rhi = float(all_r.min()), float(all_r.max())
    dv = max(vhi - vlo, 1e-4)
    dr = max(rhi - rlo, 1e-4)
    x_range = [vlo - 0.04*dv, vhi + 0.08*dv]
    y_range = [rlo - 0.06*dr, rhi + 0.12*dr]

    fig = go.Figure()

    # 1. Monte Carlo cloud
    fig.add_trace(go.Scattergl(
        x    = mc_v * 100,
        y    = mc_r * 100,
        mode = "markers",
        name = "Monte Carlo Portfolios",
        marker = dict(
            color=mc_s, colorscale="plasma", opacity=0.28, size=3,
            colorbar=dict(
                title=dict(text="Sharpe Ratio",
                           font=dict(color="#555577", size=9, family="monospace"),
                           side="right"),
                tickfont=dict(color="#555577", size=8, family="monospace"),
                outlinecolor="#1a1a3a", outlinewidth=1,
                thickness=12, len=0.72, x=1.01,
            ),
            showscale=True, line=dict(width=0),
        ),
        hovertemplate=(
            "<b>Monte Carlo Portfolio</b><br>"
            "Return     : %{y:.2f}%<br>"
            "Volatility : %{x:.2f}%<br>"
            "Sharpe     : %{marker.color:.3f}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # 2. Individual asset dots
    for i, name in enumerate(names):
        colour = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[av[i]*100], y=[ar[i]*100],
            mode="markers+text",
            name=name,
            marker=dict(color=colour, size=9, opacity=0.88,
                        line=dict(color="rgba(255,255,255,0.10)", width=0.8)),
            text=[name],
            textposition="top center",
            textfont=dict(color=colour, size=7.5, family="monospace"),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Return     : %{y:.2f}%<br>"
                "Volatility : %{x:.2f}%"
                "<extra></extra>"
            ),
            showlegend=True,
        ))

    # 3. Risk-free rate line
    fig.add_shape(type="line",
        x0=x_range[0], x1=x_range[1], y0=rf*100, y1=rf*100,
        line=dict(color="#2a2a50", width=1.0, dash="dot"), layer="below")
    fig.add_annotation(
        x=x_range[0]+0.015*dv, y=rf*100+0.022*dr,
        text=f"Rf = {rf*100:.2f}%", showarrow=False,
        font=dict(color="#333355", size=9, family="monospace"), xanchor="left")

    # 4. Max Sharpe crosshairs
    for cfg in [
        dict(x0=vs*100, x1=vs*100, y0=y_range[0], y1=rs*100),
        dict(x0=x_range[0], x1=vs*100, y0=rs*100, y1=rs*100),
    ]:
        fig.add_shape(type="line",
            line=dict(color="#8B1A1A", width=0.9, dash="dash"), layer="below", **cfg)

    # 5. Max Sharpe marker
    fig.add_trace(go.Scatter(
        x=[vs*100], y=[rs*100], mode="markers",
        name=f"MAX SHARPE  SR={ss:.3f}",
        marker=dict(symbol="circle", color="#EF5350", size=16,
                    line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate=(
            "<b>Max Sharpe Portfolio</b><br>"
            f"Return     : {rs*100:.2f}%<br>"
            f"Volatility : {vs*100:.2f}%<br>"
            f"Sharpe     : {ss:.3f}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    # 6. Min Variance crosshairs
    for cfg in [
        dict(x0=vm*100, x1=vm*100, y0=y_range[0], y1=rm*100),
        dict(x0=x_range[0], x1=vm*100, y0=rm*100, y1=rm*100),
    ]:
        fig.add_shape(type="line",
            line=dict(color="#005a50", width=0.9, dash="dash"), layer="below", **cfg)

    # 7. Min Variance marker
    fig.add_trace(go.Scatter(
        x=[vm*100], y=[rm*100], mode="markers",
        name=f"MIN VARIANCE  SR={sm:.3f}",
        marker=dict(symbol="diamond", color="#26A69A", size=14,
                    line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate=(
            "<b>Min Variance Portfolio</b><br>"
            f"Return     : {rm*100:.2f}%<br>"
            f"Volatility : {vm*100:.2f}%<br>"
            f"Sharpe     : {sm:.3f}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    # 8. Layout
    data_tag = (f"SYNTHETIC MVN  [{lookback_label} REQUESTED]" if is_synth
                else f"LIVE DATA  |  LOOKBACK: {lookback_label.upper()}")
    title_col = "#8B6914" if is_synth else "#333355"

    fig.update_layout(
        plot_bgcolor="#08080f", paper_bgcolor="#08080f",
        height=600,
        margin=dict(l=65, r=85, t=70, b=65),
        title=dict(
            text=(f"MARKOWITZ EFFICIENT FRONTIER  |  {len(names)} ASSETS  |  "
                  f"{data_tag}  |  {N_SIM:,} MC PATHS"),
            font=dict(color=title_col, size=10, family="monospace"),
            x=0.0, xanchor="left", pad=dict(l=4, b=6),
        ),
        xaxis=dict(
            title=dict(text="ANNUALISED VOLATILITY (%)",
                       font=dict(color="#333355", size=9, family="monospace"), standoff=10),
            tickfont=dict(color="#333355", size=8, family="monospace"),
            gridcolor="#101025", gridwidth=0.5,
            zerolinecolor="#1a1a3a", linecolor="#10102a",
            range=x_range, showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="ANNUALISED RETURN (%)",
                       font=dict(color="#333355", size=9, family="monospace"), standoff=10),
            tickfont=dict(color="#333355", size=8, family="monospace"),
            gridcolor="#101025", gridwidth=0.5,
            zerolinecolor="#1a1a3a", linecolor="#10102a",
            range=y_range, showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(8,8,15,0.80)", bordercolor="#14143a", borderwidth=1,
            font=dict(color="#666688", size=8.5, family="monospace"),
            x=0.01, y=0.01, xanchor="left", yanchor="bottom", itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="#0c0c20", bordercolor="#22226a",
            font=dict(color="#c0c4ff", size=11, family="monospace"),
        ),
        hovermode="closest",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  §5  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  5.1  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Identity wordmark
    st.markdown(
        "<div style='padding:10px 0 6px 0;'>"
        "<div style='font-size:0.85rem; font-weight:700; color:#5050aa; "
        "letter-spacing:0.04em;'>KUNAL KAUSHAL</div>"
        "<div style='font-size:0.62rem; color:#28285a; letter-spacing:0.1em; "
        "text-transform:uppercase; font-family:monospace;'>"
        "Portfolio Analytics  |  v3.0</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Step 1: Asset Selection ──────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; color:#44446a; "
        "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;'>"
        "Step 1 — Select Assets</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Pick 2 to {len(NIFTY_SECTOR_POOL)} stocks from our NSE universe. "
        "The optimizer will find the best combination for you."
    )
    selected_tickers: list[str] = st.multiselect(
        label="Select Assets",
        options=NIFTY_SECTOR_POOL,
        default=DEFAULT_TICKERS,
        help="Feeds into the SLSQP optimiser, Monte Carlo engine, and VaR/CVaR.",
        label_visibility="collapsed",
    )

    if selected_tickers:
        sectors    = sorted({_SECTOR_MAP.get(t, "Other") for t in selected_tickers})
        badge_html = " ".join(f'<span class="sector-pill">{s}</span>' for s in sectors)
        st.markdown(
            f"<div style='margin-top:4px;margin-bottom:2px;'>{badge_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"**{len(selected_tickers)} stock{'s' if len(selected_tickers)!=1 else ''}** "
            f"across {len(sectors)} sector{'s' if len(sectors)!=1 else ''}"
        )

    with st.expander("Add custom tickers (optional)"):
        st.caption(
            "One per line or comma-separated.  \n"
            "Bare names like `WIPRO` auto-become `WIPRO.NS`.  \n"
            "Non-NSE symbols like `AAPL` pass through unchanged."
        )
        custom_raw = st.text_area(
            label="Custom tickers", value="", height=75,
            placeholder="e.g. WIPRO\nAAPL", label_visibility="collapsed",
        )
        custom_tickers = _repair_tickers(custom_raw) if custom_raw.strip() else []
        if custom_tickers:
            st.caption("**Parsed:** " + "  ·  ".join(custom_tickers))

    seen_m: dict[str, None] = {}
    tickers: list[str]      = []
    for t in (selected_tickers + custom_tickers):
        if t not in seen_m:
            seen_m[t] = None
            tickers.append(t)

    st.markdown("---")

    # ── Advanced Settings (collapsed by default) ─────────────────────────────
    with st.expander("Advanced Settings", expanded=False):
        st.markdown(
            "<div style='font-size:0.66rem; color:#44446a; letter-spacing:0.08em; "
            "text-transform:uppercase; margin-bottom:8px;'>Historical Lookback</div>",
            unsafe_allow_html=True,
        )
        st.caption("How far back the system looks to estimate returns and risk.")
        lookback_label: str = st.selectbox(
            label="Lookback Period",
            options=list(LOOKBACK_OPTIONS.keys()),
            index=list(LOOKBACK_OPTIONS.keys()).index(LOOKBACK_DEFAULT),
            help="Short windows (<= 1 Month) trigger the Synthetic MVN fallback.",
            label_visibility="collapsed",
        )
        lookback_days: int = LOOKBACK_OPTIONS[lookback_label]

        if lookback_days <= 30:
            st.markdown(
                "<div class='lookback-warn'><b>Short window.</b>  "
                "Synthetic engine will activate automatically.</div>",
                unsafe_allow_html=True,
            )
        elif lookback_days <= 90:
            st.markdown(
                "<div class='lookback-warn'><b>Short-medium window.</b>  "
                "Interpret results with care.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.66rem; color:#44446a; letter-spacing:0.08em; "
            "text-transform:uppercase; margin-bottom:4px;'>Risk-Free Rate (%)</div>",
            unsafe_allow_html=True,
        )
        st.caption("The return you could get with zero risk (e.g., fixed deposit rate).")
        rf_pct: float = st.slider(
            label="RF (%)", min_value=0.0, max_value=15.0,
            value=DEFAULT_RF_PCT, step=0.25, format="%.2f%%",
            label_visibility="collapsed",
        )
        st.caption(f"Using **{rf_pct:.2f}%** as the risk-free benchmark.")
        rf = rf_pct / 100.0

    st.markdown("---")

    # ── Step 2: Capital ──────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; color:#44446a; "
        "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;'>"
        "Step 2 — Your Investment Capital</div>",
        unsafe_allow_html=True,
    )
    st.caption("How much money are you looking to invest? We will show you the exact rupee split.")
    total_capital: float = float(
        st.slider(
            label="Capital", min_value=1_000, max_value=100_000,
            value=100_000, step=1_000, format="Rs. %d",
            label_visibility="collapsed",
            help="Used to compute per-stock rupee allocations.",
        )
    )
    st.caption(f"Investing: **Rs. {total_capital:,.0f}**")

    st.markdown("---")

    # ── Step 3: Run ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.68rem; font-weight:700; color:#44446a; "
        "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;'>"
        "Step 3 — Run the Optimizer</div>",
        unsafe_allow_html=True,
    )
    run = st.button("Run Optimization", use_container_width=True,
                    icon=":material/play_arrow:")
    st.markdown("---")

    st.caption("Data    : yfinance  |  " + LOOKBACK_OPTIONS.get(
        locals().get("lookback_label", LOOKBACK_DEFAULT), LOOKBACK_DEFAULT).__class__.__name__)
    st.caption("Fallbk  : Synthetic MVN (756 days)")
    st.caption("Optim   : SciPy SLSQP  |  long-only")
    st.caption(f"Sim     : {N_SIM:,} Monte Carlo portfolios")
    st.caption("Risk    : 95% Historical VaR & CVaR")
    st.caption("Cov     : Tikhonov regularised (lam=1e-6)")


# Ensure lookback_label and rf are always defined (in case expander was not opened)
try:
    _ = lookback_label
except NameError:
    lookback_label = LOOKBACK_DEFAULT
    lookback_days  = LOOKBACK_OPTIONS[lookback_label]
try:
    _ = rf
except NameError:
    rf = DEFAULT_RF_PCT / 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  5.2  MAIN HEADER + DISCLAIMER BANNER
# ─────────────────────────────────────────────────────────────────────────────
# Top disclaimer — signals seriousness and trust
st.markdown(
    "<div class='disclaimer-banner'>"
    "For educational and informational purposes only. "
    "This tool does not constitute financial advice or an investment recommendation. "
    "Past performance is not indicative of future results. "
    "Consult a qualified financial advisor before making investment decisions."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1>Markowitz Portfolio Optimization</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='wordmark'>"
    "KUNAL KAUSHAL  &nbsp;|&nbsp;  "
    "IIT Kanpur NPTEL Capstone  &nbsp;|&nbsp;  "
    "SciPy SLSQP  &nbsp;|&nbsp;  "
    "95% Historical VaR / CVaR  &nbsp;|&nbsp;  "
    "v3.0"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  5.3  INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
if len(tickers) < 2:
    st.warning(
        "Select at least **2 assets** from the sidebar to run the optimisation.",
        icon=":material/warning:",
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  5.4  SESSION-STATE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_run_key = (tuple(tickers), lookback_label)

if (
    "res" not in st.session_state
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

    with st.spinner("Building interactive chart ..."):
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
prices         = R["prices"];    lr      = R["lr"]
mu             = R["mu"];        cov     = R["cov"]
ws             = R["ws"];        wm      = R["wm"]
names          = R["names"];     rf      = R["rf"]
fig            = R["fig"];       is_synth= R["is_synth"]
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
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    st.markdown("## About This Dashboard")
    st.markdown(
        "<div class='overview-block'>"
        "Developed by <strong>Kunal Kaushal</strong>, a B.Com (Hons) graduate. "
        "This dashboard is a practical capstone implementation for the "
        "<em>Advanced Financial Analytics</em> certification via "
        "<strong>IIT Kanpur (NPTEL / Swayam)</strong>. "
        "It demonstrates the deployment of SciPy-based "
        "<strong>Sequential Least Squares Programming (SLSQP)</strong> "
        "and <strong>Monte Carlo simulations</strong> for real-world portfolio optimization."
        "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### How to Use This Tool")
        steps = [
            ("Step 1", "Select Assets", "Pick 2 or more NSE stocks from the sidebar. Mix sectors for better diversification."),
            ("Step 2", "Set Your Capital", "Move the capital slider to the amount you plan to invest."),
            ("Step 3", "Run Optimization", "Click 'Run Optimization'. The engine fetches live data and finds your best portfolios."),
            ("Step 4", "Read the Chart", "Go to Portfolio Optimization. Each dot is a possible portfolio. Hover to explore."),
            ("Step 5", "Understand Results", "Go to Investment Insights for a plain-English explanation of what the numbers mean for you."),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"<div style='display:flex; align-items:flex-start; margin-bottom:12px;'>"
                f"<div style='min-width:52px; background:#10103a; border:1px solid #22225a; "
                f"border-radius:4px; padding:3px 6px; font-size:0.62rem; font-weight:700; "
                f"color:#4444aa; letter-spacing:0.08em; text-transform:uppercase; "
                f"margin-right:12px; margin-top:1px; text-align:center; flex-shrink:0;'>{num}</div>"
                f"<div><div style='font-size:0.84rem; font-weight:600; color:#8888cc; "
                f"margin-bottom:2px;'>{title}</div>"
                f"<div style='font-size:0.78rem; color:#555577; line-height:1.5;'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Mathematical Formulation")
        st.code(
            "Portfolio Return    : R_p  = w^T · mu\n"
            "Portfolio Variance  : s2_p = w^T · Sigma · w\n"
            "Sharpe Ratio        : SR   = (R_p - R_f) / s_p\n"
            "Tikhonov Reg        : Sigma_reg = Sigma + 1e-6 * I\n"
            "SLSQP Constraints   : sum(w_i)=1,  0<=w_i<=1\n"
            "VaR (95%)           : 5th percentile of daily returns\n"
            "CVaR (95%)          : E[r | r <= VaR]",
            language="text",
        )

    with col_b:
        st.markdown("### Engine Architecture")
        arch_rows = [
            ("Optimiser",   "SciPy SLSQP — Sequential Least Squares Programming (ftol=1e-12, maxiter=2,000)"),
            ("Simulation",  "Monte Carlo — 5,000 random portfolios to map the full feasible set"),
            ("Risk Engine", "95% Historical VaR & CVaR computed on actual log-return distributions"),
            ("Covariance",  "Tikhonov (ridge) regularisation lambda=1e-6 — guarantees positive-definiteness"),
            ("Fallback",    "Synthetic MVN engine — 756 days of simulated data when live data is unavailable"),
            ("Universe",    "55 NSE tickers — Nifty 50 + high-momentum picks across 12 sectors"),
            ("Cache",       "Live data cached for 1 hour via st.cache_data — fast re-runs on same session"),
        ]
        for label, desc in arch_rows:
            st.markdown(
                f"<div style='display:flex; align-items:flex-start; margin-bottom:10px; "
                f"padding-bottom:10px; border-bottom:1px solid #10103a;'>"
                f"<div style='min-width:90px; font-size:0.68rem; font-weight:700; "
                f"color:#3a3a88; letter-spacing:0.06em; text-transform:uppercase; "
                f"margin-right:14px; flex-shrink:0; padding-top:1px;'>{label}</div>"
                f"<div style='font-size:0.78rem; color:#555577; line-height:1.55;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Key Design Decisions")
        decisions = [
            ("Colorblind-safe palette", "Max Sharpe uses crimson; Min Variance uses teal — distinguishable without relying on red/green alone"),
            ("Export-ready tables",     "All allocation tables have a Download CSV button for sharing results"),
            ("Offline resilience",      "Synthetic MVN fallback ensures the app always works, even without internet"),
            ("Capital in Rupees",       "Every result is shown in both % and Rs. so the numbers feel real"),
        ]
        for title, desc in decisions:
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<span style='font-size:0.78rem; font-weight:600; color:#6666aa;'>{title}</span>"
                f"<span style='font-size:0.78rem; color:#333355;'> — {desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#18183a; font-size:0.68rem; "
        "font-family:monospace; letter-spacing:0.06em;'>"
        "MARKOWITZ MVO  |  SCIPY SLSQP  |  95% HISTORICAL VAR/CVAR  |  "
        "TIKHONOV REGULARISATION  |  YFINANCE  |  "
        "KUNAL KAUSHAL  |  IIT KANPUR NPTEL  |  v3.0"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_terminal:

    # Data source banner
    if is_synth:
        st.info(
            f"**Simulated Data Mode** — the '{lookback_label}' window returned "
            "insufficient live data. All metrics use **756 days of synthetic "
            "multivariate-normal log-returns**. Optimisation math is identical.",
            icon=":material/science:",
        )
    else:
        st.success(
            f"**Live NSE Data** — {len(prices):,} clean trading days  |  "
            f"Lookback: **{lookback_label}**",
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

    # ── Decision Aid — most important UX addition ────────────────────────────
    st.markdown("## Which Portfolio Should I Choose?")
    st.markdown(
        "<div class='decision-box'>"
        "<div class='decision-box-title'>Recommendation Guide — Based on Your Goals</div>",
        unsafe_allow_html=True,
    )

    # Determine risk label for each portfolio
    def _risk_label(vol_pct: float) -> tuple[str, str]:
        if vol_pct < 15:
            return "Low Risk", "risk-bar-fill-low"
        elif vol_pct < 25:
            return "Medium Risk", "risk-bar-fill-mid"
        else:
            return "High Risk", "risk-bar-fill-high"

    s_risk_label, s_risk_class = _risk_label(vs*100)
    m_risk_label, m_risk_class = _risk_label(vm*100)
    bar_width_s = min(100, int(vs * 250))
    bar_width_m = min(100, int(vm * 250))

    da1, da2 = st.columns(2, gap="large")
    with da1:
        st.markdown(
            "<div class='decision-pill-growth'>"
            "<div style='font-size:0.65rem; font-weight:700; color:#6644aa; "
            "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;'>"
            "If you want Growth</div>"
            "<div style='font-size:0.82rem; color:#9988cc; line-height:1.5;'>"
            "Choose the <strong style='color:#EF5350;'>Max Sharpe Portfolio</strong>. "
            "It gives you the best return for every unit of risk you take on. "
            "Ideal for long-term investors comfortable with some ups and downs."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.68rem; color:#44446a; margin-top:6px;'>"
            f"Expected Return: <strong style='color:#EF5350;'>{rs*100:.1f}%</strong> "
            f"&nbsp;|&nbsp; Risk: <strong style='color:#EF5350;'>{vs*100:.1f}%</strong> "
            f"&nbsp;|&nbsp; {s_risk_label}"
            f"</div>"
            f"<div class='risk-bar-wrap'><div class='{s_risk_class}' "
            f"style='width:{bar_width_s}%;'></div></div>",
            unsafe_allow_html=True,
        )
    with da2:
        st.markdown(
            "<div class='decision-pill-safety'>"
            "<div style='font-size:0.65rem; font-weight:700; color:#226655; "
            "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;'>"
            "If you want Safety</div>"
            "<div style='font-size:0.82rem; color:#669988; line-height:1.5;'>"
            "Choose the <strong style='color:#26A69A;'>Min Variance Portfolio</strong>. "
            "It minimises how much your money fluctuates day to day. "
            "Ideal for conservative investors or shorter time horizons."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.68rem; color:#224433; margin-top:6px;'>"
            f"Expected Return: <strong style='color:#26A69A;'>{rm*100:.1f}%</strong> "
            f"&nbsp;|&nbsp; Risk: <strong style='color:#26A69A;'>{vm*100:.1f}%</strong> "
            f"&nbsp;|&nbsp; {m_risk_label}"
            f"</div>"
            f"<div class='risk-bar-wrap'><div class='{m_risk_class}' "
            f"style='width:{bar_width_m}%;'></div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Efficient Frontier Chart ─────────────────────────────────────────────
    st.markdown("## Efficient Frontier")

    # Plain-English chart caption — directly above chart
    st.markdown(
        "<div class='chart-caption'>"
        "<strong style='color:#5555aa;'>How to read this chart:</strong>  "
        "Every dot represents one possible way to combine your selected stocks into a portfolio. "
        "Move right = more risk. Move up = better return. "
        "The <strong style='color:#EF5350;'>red circle</strong> is your mathematically best "
        "risk-adjusted portfolio (Max Sharpe). "
        "The <strong style='color:#26A69A;'>teal diamond</strong> is the safest option (Min Variance). "
        "Hover over any dot for exact numbers."
        "</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # How to read expander
    with st.expander("Understanding the Efficient Frontier in depth"):
        st.markdown(
            "**What is the Efficient Frontier?**  \n"
            "It is the boundary of the best possible portfolios. Any portfolio sitting "
            "on this boundary cannot be improved — you cannot get more return without "
            "also taking on more risk, and you cannot reduce risk without also reducing return.  \n\n"
            "**What are all the other dots?**  \n"
            f"The {N_SIM:,} faint coloured dots are randomly constructed portfolios "
            "generated by the Monte Carlo engine. They map out the entire 'cloud' of "
            "possible portfolios from your selected stocks. The colour of each dot "
            "shows its Sharpe ratio — brighter means a better ratio of return to risk.  \n\n"
            "**Why are the coloured dots (individual stocks) often far from the frontier?**  \n"
            "A single stock portfolio is inherently undiversified. By combining stocks "
            "intelligently, the optimizer can reach a point on the frontier that none "
            "of the individual stocks can reach alone."
        )

    st.markdown("---")

    # ── Capital Allocation Summary ───────────────────────────────────────────
    st.markdown("## Capital Allocation Summary")
    st.caption(
        f"Based on **Rs. {total_capital:,.0f}** investment capital. "
        "Stocks below 0.01% weight are hidden. "
        "Use the sidebar slider to change your capital."
    )

    col_sharpe, col_minvar = st.columns(2, gap="large")

    def _build_alloc_df(weights: np.ndarray, asset_names: list[str],
                        capital: float, min_wt: float = 0.0001) -> pd.DataFrame:
        mask      = weights >= min_wt
        tickers_f = [asset_names[i] for i in range(len(asset_names)) if mask[i]]
        weights_f = weights[mask]
        rupees_f  = weights_f * capital
        return pd.DataFrame({
            "Ticker"          : tickers_f,
            "Weight (%)"      : (weights_f * 100).astype(float),
            "Allocation (Rs.)": rupees_f.astype(float),
        }).set_index("Ticker")

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
                "Weight (%)"      : st.column_config.NumberColumn("Weight (%)", format="%.2f"),
                "Allocation (Rs.)": st.column_config.NumberColumn("Allocation (Rs.)", format="Rs. %,.0f"),
            },
        )
        # Download CSV button
        st.download_button(
            label="Download Max Sharpe Allocation (CSV)",
            data=sharpe_alloc.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="max_sharpe_allocation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"Sum of weights: **{ws.sum()*100:.4f}%**")

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
                "Weight (%)"      : st.column_config.NumberColumn("Weight (%)", format="%.2f"),
                "Allocation (Rs.)": st.column_config.NumberColumn("Allocation (Rs.)", format="Rs. %,.0f"),
            },
        )
        st.download_button(
            label="Download Min Variance Allocation (CSV)",
            data=minvar_alloc.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="min_variance_allocation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"Sum of weights: **{wm.sum()*100:.4f}%**")

    st.markdown("---")

    # ── Performance Summary Grid ─────────────────────────────────────────────
    st.markdown("## Performance Summary")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Max Sharpe Portfolio")
        a, b = st.columns(2)
        a.metric("Annual Return",     f"{rs*100:.4f}%", f"{(rs-rm)*100:+.4f}% vs MinVar")
        b.metric("Annual Volatility", f"{vs*100:.4f}%", f"{(vs-vm)*100:+.4f}% vs MinVar")
        a.metric("Sharpe Ratio",      f"{ss:.4f}",      f"{ss-sm:+.4f} vs MinVar")
        b.metric("Risk-Free Rate",    f"{rf*100:.2f}%")
        a.metric("95% Daily VaR",     f"{vars_*100:.4f}%", f"{(vars_-varm)*100:+.4f}% vs MinVar")
        b.metric("95% Daily CVaR",    f"{cvars*100:.4f}%", f"{(cvars-cvarm)*100:+.4f}% vs MinVar")

    with right:
        st.markdown("### Min Variance Portfolio")
        c, d = st.columns(2)
        c.metric("Annual Return",     f"{rm*100:.4f}%", f"{(rm-rs)*100:+.4f}% vs MaxSR")
        d.metric("Annual Volatility", f"{vm*100:.4f}%", f"{(vm-vs)*100:+.4f}% vs MaxSR")
        c.metric("Sharpe Ratio",      f"{sm:.4f}",      f"{sm-ss:+.4f} vs MaxSR")
        d.metric("Risk-Free Rate",    f"{rf*100:.2f}%")
        c.metric("95% Daily VaR",     f"{varm*100:.4f}%", f"{(varm-vars_)*100:+.4f}% vs MaxSR")
        d.metric("95% Daily CVaR",    f"{cvarm*100:.4f}%", f"{(cvarm-cvars)*100:+.4f}% vs MaxSR")

    st.markdown("---")

    # ── Core Metrics Table ───────────────────────────────────────────────────
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
        metrics_df, use_container_width=True, height=260,
        column_config={
            "Max Sharpe"  : st.column_config.NumberColumn("Max Sharpe",   format="%.4f"),
            "Min Variance": st.column_config.NumberColumn("Min Variance", format="%.4f"),
        },
    )
    st.markdown("---")

    # ── Asset Allocation Detail ───────────────────────────────────────────────
    st.markdown("## Full Asset Allocation Detail")

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
        alloc_df, use_container_width=True,
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

    # ── Individual Asset Risk ─────────────────────────────────────────────────
    st.markdown("## Individual Asset Risk Metrics")

    _var_l, _cvar_l = [], []
    for i in range(len(names)):
        w_s = np.zeros(len(names)); w_s[i] = 1.0
        v, cv = hist_var_cvar(w_s, lr)
        _var_l.append(float(v * 100))
        _cvar_l.append(float(cv * 100))

    asset_risk_df = pd.DataFrame(
        {
            "Annual Return (%)" : (ar * 100).astype(float),
            "Annual Vol (%)"    : (av * 100).astype(float),
            "95% Daily VaR (%)" : _var_l,
            "95% Daily CVaR (%)": _cvar_l,
        },
        index=pd.Index(names, name="Ticker"),
        dtype=float,
    )
    st.dataframe(
        asset_risk_df, use_container_width=True,
        height=(len(names) + 2) * 38,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4f")
            for col in asset_risk_df.columns
        },
    )
    st.markdown("---")

    st.markdown(
        "<p style='text-align:center; color:#10103a; font-size:0.68rem; "
        "font-family:monospace; letter-spacing:0.05em;'>"
        "MARKOWITZ MVO  |  SCIPY SLSQP  |  95% HISTORICAL VAR/CVAR  |  "
        f"LOOKBACK: {lookback_label.upper()}  |  NIFTY 50 + HIGH-MOMENTUM (55 TICKERS)  |  "
        "TIKHONOV REGULARISATION  |  YFINANCE  |  v3.0"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — INVESTMENT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:

    st.markdown("## Investment Insights")
    st.caption(
        "Your optimization results, translated into plain English. "
        "No finance jargon. Just what the numbers actually mean for your money."
    )

    if "res" not in st.session_state:
        st.info("Run the optimization first.", icon=":material/info:")
        st.stop()

    # Pre-compute values
    sharpe_sorted = sorted(zip(names, ws), key=lambda x: x[1], reverse=True)
    minvar_sorted = sorted(zip(names, wm), key=lambda x: x[1], reverse=True)
    top_sharpe    = [(n, w) for n, w in sharpe_sorted if w > 0.0001][:3]
    top_minvar    = [(n, w) for n, w in minvar_sorted if w > 0.0001][:3]
    dominant_sharpe = top_sharpe[0][0] if top_sharpe else names[0]
    top2_names_s    = ", ".join(n for n, _ in top_sharpe[:2])
    cap_rs_inr   = total_capital * rs
    cap_rm_inr   = total_capital * rm
    cap_var_s    = abs(vars_) * total_capital
    cap_var_m    = abs(varm)  * total_capital
    cap_cvar_s   = abs(cvars) * total_capital

    # ── Quick Comparison Card at the top ────────────────────────────────────
    st.markdown("### Side-by-Side Comparison")
    st.markdown(
        "<div class='decision-box'>",
        unsafe_allow_html=True,
    )
    cmp1, cmp_div, cmp2 = st.columns([5, 1, 5])

    with cmp1:
        st.markdown(
            "<div style='font-size:0.65rem; font-weight:700; color:#cc4444; "
            "letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px;'>"
            "Max Sharpe Portfolio</div>",
            unsafe_allow_html=True,
        )
        bullets_s = [
            (f"Expected annual return: {rs*100:.1f}%", "growth"),
            (f"Annual risk (volatility): {vs*100:.1f}%", "risk"),
            (f"Sharpe ratio: {ss:.2f}", "quality"),
            (f"Max daily loss (95% VaR): Rs. {cap_var_s:,.0f}", "loss"),
        ]
        for text, _ in bullets_s:
            st.markdown(
                f"<div style='font-size:0.80rem; color:#aa6666; margin-bottom:6px; "
                f"padding-left:10px; border-left:2px solid #441414;'>{text}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='font-size:0.72rem; color:#663333; margin-top:8px;'>"
            "Best for: Long-term growth investors</div>",
            unsafe_allow_html=True,
        )

    with cmp_div:
        st.markdown(
            "<div style='width:1px; background:#18183a; height:180px; "
            "margin:0 auto; margin-top:20px;'></div>",
            unsafe_allow_html=True,
        )

    with cmp2:
        st.markdown(
            "<div style='font-size:0.65rem; font-weight:700; color:#2a8a80; "
            "letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px;'>"
            "Min Variance Portfolio</div>",
            unsafe_allow_html=True,
        )
        bullets_m = [
            (f"Expected annual return: {rm*100:.1f}%", "growth"),
            (f"Annual risk (volatility): {vm*100:.1f}%", "risk"),
            (f"Sharpe ratio: {sm:.2f}", "quality"),
            (f"Max daily loss (95% VaR): Rs. {cap_var_m:,.0f}", "loss"),
        ]
        for text, _ in bullets_m:
            st.markdown(
                f"<div style='font-size:0.80rem; color:#448877; margin-bottom:6px; "
                f"padding-left:10px; border-left:2px solid #0a2a22;'>{text}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='font-size:0.72rem; color:#224433; margin-top:8px;'>"
            "Best for: Conservative / short-horizon investors</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Section A — Max Sharpe in plain English ──────────────────────────────
    st.markdown("### A.   What Your Best Balance Portfolio Means")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Max Sharpe Portfolio — Best Risk-Adjusted Balance</h4>",
            unsafe_allow_html=True,
        )
        # Pull quote
        st.markdown(
            f"<div class='pull-quote'>Rs. {total_capital + cap_rs_inr:,.0f}</div>"
            f"<div class='pull-quote-label'>Projected portfolio value after 1 year "
            f"(Rs. {total_capital:,.0f} invested at {rs*100:.1f}% expected return)</div>",
            unsafe_allow_html=True,
        )

        ra1, ra2, ra3 = st.columns(3)
        ra1.metric("Expected Annual Return",   f"{rs*100:.2f}%")
        ra2.metric("Annual Risk (Volatility)", f"{vs*100:.2f}%")
        ra3.metric("Sharpe Ratio",             f"{ss:.3f}")

        if rs > 0:
            narrative_a = (
                f"If you invest **Rs. {total_capital:,.0f}** into this portfolio, "
                f"the optimizer — working from {lookback_label.lower()} of "
                f"{'simulated' if is_synth else 'real NSE'} market data — projects "
                f"an expected annual gain of approximately **Rs. {cap_rs_inr:,.0f}** "
                f"({rs*100:.1f}% return). That would grow your investment to roughly "
                f"**Rs. {total_capital + cap_rs_inr:,.0f}** in one year.\n\n"
                f"This is not a guarantee. The annual volatility of **{vs*100:.1f}%** "
                f"means the actual outcome could be better or worse. Think of volatility "
                f"as the normal range of ups and downs throughout the year.\n\n"
                f"The **Sharpe Ratio of {ss:.2f}** means you earn {ss:.2f} units of "
                f"excess return for every unit of risk you take. "
                f"{'Above 1.0 is generally considered good.' if ss > 1.0 else 'Below 1.0 means the return may not fully compensate for the risk taken.'}"
            )
        else:
            narrative_a = (
                f"This portfolio shows a negative expected return of **{rs*100:.2f}%** "
                f"for the {lookback_label.lower()} window selected. This may reflect a "
                f"difficult market period in the data. Consider extending the lookback "
                f"period or changing the stock selection."
            )
        st.markdown(narrative_a)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section B — Min Variance in plain English ────────────────────────────
    st.markdown("### B.   What the Safest Portfolio Means")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Min Variance Portfolio — Lowest Achievable Risk</h4>",
            unsafe_allow_html=True,
        )
        # Pull quote
        st.markdown(
            f"<div class='pull-quote'>{vm*100:.1f}%</div>"
            f"<div class='pull-quote-label'>Annual volatility — the lowest achievable "
            f"risk with your selected stocks</div>",
            unsafe_allow_html=True,
        )

        rb1, rb2, rb3 = st.columns(3)
        rb1.metric("Expected Annual Return",   f"{rm*100:.2f}%")
        rb2.metric("Annual Risk (Volatility)", f"{vm*100:.2f}%")
        rb3.metric("Sharpe Ratio",             f"{sm:.3f}")

        vol_diff = (vs - vm) * 100
        narrative_b = (
            f"The Safest Portfolio is built for one purpose: "
            f"**minimise how much your portfolio swings in value**. "
            f"It achieves a volatility of **{vm*100:.1f}%** — "
            f"**{abs(vol_diff):.1f} percentage points "
            f"{'lower' if vol_diff > 0 else 'higher'} than the Max Sharpe portfolio** "
            f"({vs*100:.1f}%).\n\n"
            f"The trade-off is return. This portfolio targets **{rm*100:.1f}% annually** "
            f"versus {rs*100:.1f}% for Max Sharpe. On Rs. {total_capital:,.0f}, "
            f"that is a difference of roughly **Rs. {abs(cap_rs_inr - cap_rm_inr):,.0f} "
            f"per year** in expected gain.\n\n"
            f"**Who should choose this?** Conservative investors, people closer to "
            f"needing the money, or anyone who cannot handle seeing large daily swings "
            f"in their portfolio value. You give up some upside, but you significantly "
            f"reduce the chance of a painful drawdown."
        )
        st.markdown(narrative_b)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section C — VaR in Rupees ─────────────────────────────────────────────
    st.markdown("### C.   Your Worst-Case Daily Loss in Actual Rupees")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>95% Historical Value at Risk — Daily Loss Threshold</h4>",
            unsafe_allow_html=True,
        )
        # Pull quote
        st.markdown(
            f"<div class='pull-quote'>Rs. {cap_var_s:,.0f}</div>"
            f"<div class='pull-quote-label'>Max Sharpe worst-case single-day loss "
            f"(95% confidence, based on Rs. {total_capital:,.0f} invested)</div>",
            unsafe_allow_html=True,
        )

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Max Sharpe VaR (%)",    f"{vars_*100:.3f}%")
        rc2.metric("Max Sharpe VaR (Rs.)",  f"Rs. {cap_var_s:,.0f}")
        rc3.metric("Min Variance VaR (%)",  f"{varm*100:.3f}%")
        rc4.metric("Min Variance VaR (Rs.)",f"Rs. {cap_var_m:,.0f}")

        narrative_c = (
            f"**Value at Risk (VaR)** answers: *How much could I lose on a really bad day?*\n\n"
            f"For your **Rs. {total_capital:,.0f}** in the Max Sharpe portfolio, "
            f"the 95% daily VaR is **Rs. {cap_var_s:,.0f}** ({abs(vars_*100):.3f}%). "
            f"On **95 out of every 100 trading days**, your single-day loss will NOT "
            f"exceed this amount. The remaining 5 days per 100 — roughly 12 to 13 days "
            f"a year — could see losses beyond this.\n\n"
            f"The Min Variance portfolio's daily VaR is **Rs. {cap_var_m:,.0f}** — "
            f"Rs. {abs(cap_var_s - cap_var_m):,.0f} "
            f"{'less' if cap_var_m < cap_var_s else 'more'} per day, confirming it "
            f"is genuinely safer on a day-to-day basis too.\n\n"
            f"The CVaR of Rs. {cap_cvar_s:,.0f} is what you would expect to lose "
            f"*on average* on those rare worst-case days beyond the VaR threshold. "
            f"This is the figure professional risk managers use for stress-testing."
        )
        st.markdown(narrative_c)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section D — Why Concentration ────────────────────────────────────────
    st.markdown("### D.   Why Certain Stocks Dominate the Allocation")

    with st.container():
        st.markdown(
            "<div class='report-card'>"
            "<h4>Understanding Concentration — The SLSQP Optimizer's Logic</h4>",
            unsafe_allow_html=True,
        )

        rd1, rd2 = st.columns(2)
        with rd1:
            st.markdown(
                "<div style='font-size:0.68rem; font-weight:700; color:#cc4444; "
                "letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;'>"
                "Max Sharpe — Top Holdings</div>",
                unsafe_allow_html=True,
            )
            for i, (nm, wt) in enumerate(top_sharpe, 1):
                # Progress bar showing weight
                bar_w = min(100, int(wt * 400))
                st.markdown(
                    f"<div style='margin-bottom:8px;'>"
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.78rem; margin-bottom:3px;'>"
                    f"<span style='color:#8888cc; font-family:monospace;'>#{i} {nm}</span>"
                    f"<span style='color:#EF5350; font-weight:700;'>"
                    f"{wt*100:.1f}%  —  Rs. {wt*total_capital:,.0f}</span></div>"
                    f"<div class='risk-bar-wrap'><div style='height:4px; border-radius:4px; "
                    f"background:#EF5350; width:{bar_w}%;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with rd2:
            st.markdown(
                "<div style='font-size:0.68rem; font-weight:700; color:#2a8a80; "
                "letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;'>"
                "Min Variance — Top Holdings</div>",
                unsafe_allow_html=True,
            )
            for i, (nm, wt) in enumerate(top_minvar, 1):
                bar_w = min(100, int(wt * 400))
                st.markdown(
                    f"<div style='margin-bottom:8px;'>"
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.78rem; margin-bottom:3px;'>"
                    f"<span style='color:#8888cc; font-family:monospace;'>#{i} {nm}</span>"
                    f"<span style='color:#26A69A; font-weight:700;'>"
                    f"{wt*100:.1f}%  —  Rs. {wt*total_capital:,.0f}</span></div>"
                    f"<div class='risk-bar-wrap'><div style='height:4px; border-radius:4px; "
                    f"background:#26A69A; width:{bar_w}%;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        narrative_d = (
            f"You might wonder: *why is so much money concentrated in just a few stocks?*\n\n"
            f"The SLSQP optimizer solves a precise mathematical problem: *given the historical "
            f"returns and correlations of all {len(names)} stocks, find the exact weights that "
            f"maximise the Sharpe ratio.* It does not care about equal distribution — it cares "
            f"about efficiency.\n\n"
            f"**{top2_names_s}** dominate the Max Sharpe portfolio because, over the last "
            f"{lookback_label.lower()}, they offered the best combination of high individual "
            f"return AND low correlation with each other. When two stocks move in different "
            f"directions, holding both simultaneously reduces total portfolio risk without "
            f"sacrificing proportionate return — this is the mathematical heart of "
            f"Markowitz diversification.\n\n"
            f"Stocks that received near-zero weights either had lower risk-adjusted returns, "
            f"or they were too correlated with the dominant holdings — adding them would have "
            f"increased risk without meaningfully increasing return.\n\n"
            f"**Important:** These allocations are based on *historical* data from the "
            f"{lookback_label.lower()} window. Past correlations do not guarantee future "
            f"performance. Longer lookback periods generally produce more stable estimates."
        )
        st.markdown(narrative_d)
        st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#14143a; font-size:0.68rem; "
        "font-family:monospace; letter-spacing:0.05em;'>"
        "FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY  |  "
        "NOT FINANCIAL ADVICE  |  NOT AN INVESTMENT RECOMMENDATION  |  "
        "PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE RESULTS  |  "
        "CONSULT A QUALIFIED FINANCIAL ADVISOR BEFORE INVESTING"
        "</p>",
        unsafe_allow_html=True,
    )
