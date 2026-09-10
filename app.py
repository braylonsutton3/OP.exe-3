from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# ============================================================
# OP.exe — HIGH-CONFIDENCE FULL REBUILD
# Primary goal: live-data LONG / SHORT / WAIT calculated independently for each timeframe
#
# Included concepts:
# - 1m / 5m / 30m / 4h / 1d / 1w
# - configurable 09:00–09:30 ET or 09:30–10:00 ET ORB
# - SMA / EMA trend
# - RSI momentum state
# - ADX trend strength
# - VWAP (intraday)
# - ATR volatility
# - volatility regime
# - volume expansion
# - swing highs / lows
# - prior day / prior week highs and lows
# - session highs and lows
# - Break of Structure (BOS)
# - Change of Character / structure shift (CHoCH)
# - liquidity sweeps
# - displacement candles
# - Fair Value Gaps (FVGs)
# - objective order-block proxy
# - structural TP / SL
# - reward:risk
# - projected dollar P&L by contract
# - $200 minimum projected-profit filter for 10 contracts
#
# IMPORTANT:
# This is a rules-based analytical tool, not a guarantee.
# ============================================================

st.set_page_config(
    page_title="OP.exe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SETTINGS
# ============================================================

TIMEFRAMES = {
    "1m":  {"px_unit": 2, "px_num": 1,  "lookback_days": 7,    "limit": 1500, "yf_period": "7d",  "yf_interval": "1m"},
    "5m":  {"px_unit": 2, "px_num": 5,  "lookback_days": 45,   "limit": 1500, "yf_period": "60d", "yf_interval": "5m"},
    "30m": {"px_unit": 2, "px_num": 30, "lookback_days": 120,  "limit": 1200, "yf_period": "60d", "yf_interval": "30m"},
    "4h":  {"px_unit": 3, "px_num": 4,  "lookback_days": 365,  "limit": 1200, "yf_period": "2y",  "yf_interval": "1h"},
    "1d":  {"px_unit": 4, "px_num": 1,  "lookback_days": 1000, "limit": 1000, "yf_period": "5y",  "yf_interval": "1d"},
    "1w":  {"px_unit": 5, "px_num": 1,  "lookback_days": 2200, "limit": 500,  "yf_period": "10y", "yf_interval": "1wk"},
}

# fallback tick specs if ProjectX metadata is unavailable
FALLBACK_SPECS = {
    "MES": {"tick_size": 0.25, "tick_value": 1.25,  "yf": "MES=F"},
    "ES":  {"tick_size": 0.25, "tick_value": 12.50, "yf": "ES=F"},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,  "yf": "MNQ=F"},
    "NQ":  {"tick_size": 0.25, "tick_value": 5.00,  "yf": "NQ=F"},
    "M2K": {"tick_size": 0.10, "tick_value": 0.50,  "yf": "M2K=F"},
    "RTY": {"tick_size": 0.10, "tick_value": 5.00,  "yf": "RTY=F"},
    "MYM": {"tick_size": 1.00, "tick_value": 0.50,  "yf": "MYM=F"},
    "YM":  {"tick_size": 1.00, "tick_value": 5.00,  "yf": "YM=F"},
    "MGC": {"tick_size": 0.10, "tick_value": 1.00,  "yf": "MGC=F"},
    "GC":  {"tick_size": 0.10, "tick_value": 10.00, "yf": "GC=F"},
    "MCL": {"tick_size": 0.01, "tick_value": 1.00,  "yf": "MCL=F"},
    "CL":  {"tick_size": 0.01, "tick_value": 10.00, "yf": "CL=F"},
}

PX_BASE = "https://api.topstepx.com/api"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1220px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        padding: 10px 12px;
    }
    .op-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .op-long { border-left: 6px solid #22c55e; }
    .op-short { border-left: 6px solid #ef4444; }
    .op-wait { border-left: 6px solid #9ca3af; }
    .muted { opacity: .72; font-size: .88rem; }
    .tiny { opacity: .68; font-size: .80rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_root(symbol):
    s = (symbol or "").upper().strip().replace("!", "")
    for root in sorted(FALLBACK_SPECS.keys(), key=len, reverse=True):
        if s.startswith(root):
            return root
    return s


def round_to_tick(price, tick_size):
    if price is None or not np.isfinite(price):
        return price
    if tick_size <= 0:
        return float(price)
    return round(round(float(price) / tick_size) * tick_size, 10)


def pnl_between(entry, exit_price, tick_size, tick_value, contracts):
    if tick_size <= 0:
        return 0.0
    ticks = abs(float(exit_price) - float(entry)) / tick_size
    return ticks * tick_value * int(contracts)


def safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume", "t": "Datetime"
    })

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.set_index("Datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ============================================================
# PROJECTX / TOPSTEPX LIVE DATA
# ============================================================

def get_topstep_credentials():
    try:
        return (
            st.secrets.get("TOPSTEP_USERNAME", ""),
            st.secrets.get("TOPSTEP_API_KEY", ""),
        )
    except Exception:
        return "", ""


@st.cache_data(ttl=20 * 60, show_spinner=False)
def px_login(username, api_key):
    if not username or not api_key:
        return None, "TopstepX credentials are not configured."

    try:
        r = requests.post(
            f"{PX_BASE}/Auth/loginKey",
            json={"userName": username, "apiKey": api_key},
            timeout=12,
        )
        r.raise_for_status()
        js = r.json()
        if not js.get("success") or not js.get("token"):
            return None, js.get("errorMessage") or "TopstepX authentication failed."
        return js["token"], None
    except Exception as e:
        return None, f"TopstepX login error: {e}"


def px_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "accept": "text/plain",
        "Content-Type": "application/json",
    }


@st.cache_data(ttl=60, show_spinner=False)
def px_find_contract(token, search_text, live=True):
    r = requests.post(
        f"{PX_BASE}/Contract/search",
        headers=px_headers(token),
        json={"searchText": search_text, "live": live},
        timeout=12,
    )
    r.raise_for_status()
    js = r.json()
    contracts = js.get("contracts") or []
    if not contracts:
        return None

    active = [c for c in contracts if c.get("activeContract")]
    pool = active or contracts
    root = clean_root(search_text)
    exactish = [
        c for c in pool
        if str(c.get("name", "")).upper().startswith(root)
    ]
    return (exactish or pool)[0]


def px_retrieve_bars(token, contract_id, timeframe, live=True):
    cfg = TIMEFRAMES[timeframe]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg["lookback_days"])

    payload = {
        "contractId": contract_id,
        "live": live,
        "startTime": start.isoformat().replace("+00:00", "Z"),
        "endTime": end.isoformat().replace("+00:00", "Z"),
        "unit": cfg["px_unit"],
        "unitNumber": cfg["px_num"],
        "limit": cfg["limit"],
        "includePartialBar": True,
    }

    r = requests.post(
        f"{PX_BASE}/History/retrieveBars",
        headers=px_headers(token),
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    return normalize_df(pd.DataFrame(js.get("bars") or []))


# ============================================================
# YAHOO FALLBACK
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def yahoo_bars(symbol, timeframe):
    root = clean_root(symbol)
    yf_symbol = FALLBACK_SPECS.get(root, {}).get("yf", symbol)
    cfg = TIMEFRAMES[timeframe]

    data = yf.download(
        yf_symbol,
        period=cfg["yf_period"],
        interval=cfg["yf_interval"],
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = normalize_df(data)

    if timeframe == "4h" and not data.empty:
        data = (
            data.resample("4h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna(subset=["Open", "High", "Low", "Close"])
        )

    return data


# ============================================================
# CORE INDICATORS
# ============================================================

def atr_series(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_atr(df, period=14):
    if df is None or len(df) < period + 2:
        return np.nan
    return safe_float(atr_series(df, period).iloc[-1])


def calculate_rsi(series, period=14):
    if len(series) < period + 2:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = (100 - (100 / (1 + rs))).iloc[-1]
    return 50.0 if not np.isfinite(value) else float(value)


def calculate_adx(df, period=14):
    if df is None or len(df) < period * 2 + 2:
        return np.nan

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smoothed = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_smoothed.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_smoothed.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return safe_float(dx.rolling(period).mean().iloc[-1])


def intraday_vwap(df):
    if df is None or df.empty:
        return np.nan
    if "Volume" not in df.columns or df["Volume"].fillna(0).sum() <= 0:
        return np.nan

    local = df.tz_convert("America/New_York")
    latest_date = local.index[-1].date()
    day = local[local.index.date == latest_date]
    if day.empty:
        return np.nan

    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    vol = day["Volume"].fillna(0)
    denom = vol.cumsum().iloc[-1]
    if denom <= 0:
        return np.nan
    return float((typical * vol).cumsum().iloc[-1] / denom)


def volume_expansion(df, lookback=20):
    if df is None or len(df) < lookback + 2:
        return 1.0
    vol = df["Volume"].fillna(0)
    baseline = vol.iloc[-lookback-1:-1].mean()
    if baseline <= 0:
        return 1.0
    return float(vol.iloc[-1] / baseline)


def volatility_regime(df):
    if df is None or len(df) < 60:
        return "UNKNOWN", 1.0

    atr14 = atr_series(df, 14)
    current = safe_float(atr14.iloc[-1])
    baseline = safe_float(atr14.iloc[-50:].median())

    if not np.isfinite(current) or not np.isfinite(baseline) or baseline <= 0:
        return "UNKNOWN", 1.0

    ratio = current / baseline

    if ratio >= 1.35:
        return "EXPANSION", ratio
    if ratio <= 0.75:
        return "COMPRESSION", ratio
    return "NORMAL", ratio


# ============================================================
# SWING STRUCTURE
# ============================================================

def swing_levels(df, window=3, lookback=180):
    data = df.tail(lookback)
    highs, lows = [], []

    if len(data) < window * 2 + 2:
        return highs, lows

    hv = data["High"].to_numpy()
    lv = data["Low"].to_numpy()
    idx = data.index

    for i in range(window, len(data) - window):
        if hv[i] >= np.max(hv[i-window:i+window+1]):
            highs.append((idx[i], float(hv[i])))
        if lv[i] <= np.min(lv[i-window:i+window+1]):
            lows.append((idx[i], float(lv[i])))

    return highs[-20:], lows[-20:]


def detect_bos_choch(df):
    """
    Objective proxy:
    - Bullish BOS = latest close breaks above latest confirmed swing high.
    - Bearish BOS = latest close breaks below latest confirmed swing low.
    - CHoCH = break is opposite the direction implied by the last two swing legs.
    """
    highs, lows = swing_levels(df)
    if not highs or not lows:
        return {"bos": "NONE", "choch": "NONE", "broken_level": None}

    close = float(df["Close"].iloc[-1])
    last_high = highs[-1][1]
    last_low = lows[-1][1]

    # estimate prior structure direction from last two highs/lows
    structure = "NEUTRAL"
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1][1] > highs[-2][1]
        higher_low = lows[-1][1] > lows[-2][1]
        lower_high = highs[-1][1] < highs[-2][1]
        lower_low = lows[-1][1] < lows[-2][1]
        if higher_high and higher_low:
            structure = "UP"
        elif lower_high and lower_low:
            structure = "DOWN"

    bos = "NONE"
    choch = "NONE"
    level = None

    if close > last_high:
        bos = "BULLISH"
        level = last_high
        if structure == "DOWN":
            choch = "BULLISH"
    elif close < last_low:
        bos = "BEARISH"
        level = last_low
        if structure == "UP":
            choch = "BEARISH"

    return {"bos": bos, "choch": choch, "broken_level": level}


# ============================================================
# LIQUIDITY SWEEPS
# ============================================================

def detect_liquidity_sweep(df):
    """
    Bearish sweep:
    latest wick trades above a prior confirmed swing high,
    but closes back below that level.

    Bullish sweep:
    latest wick trades below a prior confirmed swing low,
    but closes back above that level.
    """
    if df is None or len(df) < 20:
        return {"type": "NONE", "level": None}

    highs, lows = swing_levels(df.iloc[:-1])
    if not highs or not lows:
        return {"type": "NONE", "level": None}

    bar = df.iloc[-1]
    prior_high = highs[-1][1]
    prior_low = lows[-1][1]

    if bar["High"] > prior_high and bar["Close"] < prior_high:
        return {"type": "BEARISH_SWEEP", "level": prior_high}

    if bar["Low"] < prior_low and bar["Close"] > prior_low:
        return {"type": "BULLISH_SWEEP", "level": prior_low}

    return {"type": "NONE", "level": None}


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(df):
    if df is None or len(df) < 20:
        return {"direction": "NONE", "strength": 0.0}

    a = calculate_atr(df)
    if not np.isfinite(a) or a <= 0:
        return {"direction": "NONE", "strength": 0.0}

    bar = df.iloc[-1]
    body = abs(float(bar["Close"] - bar["Open"]))
    full_range = max(float(bar["High"] - bar["Low"]), 1e-9)
    body_ratio = body / full_range
    strength = body / a

    if strength >= 0.80 and body_ratio >= 0.60:
        direction = "BULLISH" if bar["Close"] > bar["Open"] else "BEARISH"
        return {"direction": direction, "strength": strength}

    return {"direction": "NONE", "strength": strength}


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def recent_fvgs(df, lookback=160):
    out = []
    if df is None or len(df) < 3:
        return out

    start = max(2, len(df) - lookback)

    for i in range(start, len(df)):
        left = df.iloc[i - 2]
        current = df.iloc[i]

        # bullish 3-candle imbalance
        if current["Low"] > left["High"]:
            out.append({
                "direction": "BULL",
                "low": float(left["High"]),
                "high": float(current["Low"]),
                "time": df.index[i],
            })

        # bearish 3-candle imbalance
        if current["High"] < left["Low"]:
            out.append({
                "direction": "BEAR",
                "low": float(current["High"]),
                "high": float(left["Low"]),
                "time": df.index[i],
            })

    return out[-40:]


# ============================================================
# ORDER BLOCK PROXY
# ============================================================

def recent_order_blocks(df, lookback=80):
    """
    Objective order-block proxy:
    - bullish OB = last bearish candle immediately before a strong bullish
      displacement that closes above a recent local high
    - bearish OB = last bullish candle immediately before a strong bearish
      displacement that closes below a recent local low

    This is deliberately given LOW weight because "order block" definitions
    are not standardized and evidence is weaker than basic trend/volatility/risk.
    """
    out = []
    if df is None or len(df) < 25:
        return out

    a_series = atr_series(df, 14)
    start = max(15, len(df) - lookback)

    for i in range(start, len(df)):
        a = safe_float(a_series.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue

        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        body = abs(float(bar["Close"] - bar["Open"]))

        local_high = float(df["High"].iloc[max(0, i-8):i].max())
        local_low = float(df["Low"].iloc[max(0, i-8):i].min())

        bullish_displacement = (
            bar["Close"] > bar["Open"]
            and body >= 0.8 * a
            and bar["Close"] > local_high
        )

        bearish_displacement = (
            bar["Close"] < bar["Open"]
            and body >= 0.8 * a
            and bar["Close"] < local_low
        )

        if bullish_displacement and prev["Close"] < prev["Open"]:
            out.append({
                "direction": "BULL",
                "low": float(prev["Low"]),
                "high": float(prev["High"]),
                "time": df.index[i - 1],
            })

        if bearish_displacement and prev["Close"] > prev["Open"]:
            out.append({
                "direction": "BEAR",
                "low": float(prev["Low"]),
                "high": float(prev["High"]),
                "time": df.index[i - 1],
            })

    return out[-20:]


# ============================================================
# PRIOR DAY / WEEK / SESSION LEVELS
# ============================================================

def prior_day_week_levels(df):
    if df is None or df.empty:
        return {}

    local = df.tz_convert("America/New_York")
    daily = local.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly = local.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()

    result = {}

    if len(daily) >= 2:
        result["PDH"] = float(daily["High"].iloc[-2])
        result["PDL"] = float(daily["Low"].iloc[-2])

    if len(weekly) >= 2:
        result["PWH"] = float(weekly["High"].iloc[-2])
        result["PWL"] = float(weekly["Low"].iloc[-2])

    latest_date = local.index[-1].date()
    current_day = local[local.index.date == latest_date]
    if not current_day.empty:
        result["SESSION_HIGH"] = float(current_day["High"].max())
        result["SESSION_LOW"] = float(current_day["Low"].min())

    return result


# ============================================================
# ORB
# ============================================================

def latest_orb(df_1m, mode):
    if df_1m is None or df_1m.empty:
        return None

    if mode == "09:30–10:00 ET":
        start_t, end_t = "09:30", "09:59"
    else:
        start_t, end_t = "09:00", "09:29"

    local = df_1m.tz_convert("America/New_York")
    unique_dates = list(dict.fromkeys(local.index.date))[::-1]

    for d in unique_dates:
        day = local[local.index.date == d]
        orb = day.between_time(start_t, end_t)
        if len(orb) >= 10:
            hi = float(orb["High"].max())
            lo = float(orb["Low"].min())
            return {
                "date": d,
                "high": hi,
                "low": lo,
                "mid": (hi + lo) / 2,
                "mode": mode,
            }
    return None


# ============================================================
# EVIDENCE ENGINE
# ============================================================

def evidence_engine(df, timeframe):
    """
    Produces a directional evidence score and a 0-100 CONFLUENCE INDEX.

    IMPORTANT:
    The confluence index is NOT a probability of winning.
    It only measures how much of OP.exe's own evidence agrees in one direction.
    """
    if df is None or len(df) < 60:
        return {
            "raw": "WAIT",
            "score": 0,
            "bull": 0,
            "bear": 0,
            "confluence_index": 0.0,
            "dominance": 0.0,
            "reasons": ["Not enough data"],
            "atr": np.nan,
            "rsi": np.nan,
            "adx": np.nan,
            "regime": "UNKNOWN",
            "bos": "NONE",
            "choch": "NONE",
            "sweep": "NONE",
            "displacement": "NONE",
            "volume_ratio": 1.0,
            "vwap": np.nan,
        }

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    current = float(close.iloc[-1])
    a = calculate_atr(df)
    rv = calculate_rsi(close)
    adx = calculate_adx(df)
    regime, vol_ratio = volatility_regime(df)
    vol_exp = volume_expansion(df)
    vwap = intraday_vwap(df) if timeframe in ("1m", "5m", "30m") else np.nan
    structure = detect_bos_choch(df)
    sweep = detect_liquidity_sweep(df)
    displacement = detect_displacement(df)
    gaps = recent_fvgs(df)
    obs = recent_order_blocks(df)

    bull = 0.0
    bear = 0.0
    max_points = 0.0
    reasons = []

    def vote(direction, points, label):
        nonlocal bull, bear, max_points
        max_points += float(points)
        if direction == "BULL":
            bull += float(points)
            reasons.append(label)
        elif direction == "BEAR":
            bear += float(points)
            reasons.append(label)

    # Trend / momentum — strongest broad empirical base among these inputs
    vote("BULL" if current > sma20.iloc[-1] else "BEAR", 1.5, "price vs SMA20")
    vote("BULL" if sma20.iloc[-1] > sma50.iloc[-1] else "BEAR", 2.0, "SMA20 vs SMA50")
    vote("BULL" if ema9.iloc[-1] > ema21.iloc[-1] else "BEAR", 1.5, "EMA9 vs EMA21")

    momentum_n = min(5, len(close) - 1)
    momentum = float(close.iloc[-1] - close.iloc[-1 - momentum_n])
    if momentum > 0:
        vote("BULL", 1.5, "recent momentum up")
    elif momentum < 0:
        vote("BEAR", 1.5, "recent momentum down")
    else:
        max_points += 1.5

    if rv >= 55:
        vote("BULL", 1.0, "RSI bullish state")
    elif rv <= 45:
        vote("BEAR", 1.0, "RSI bearish state")
    else:
        max_points += 1.0

    # Intraday VWAP
    if timeframe in ("1m", "5m", "30m"):
        if np.isfinite(vwap):
            vote("BULL" if current > vwap else "BEAR", 1.5, "VWAP location")
        else:
            max_points += 1.5

    # Market structure
    if structure["bos"] == "BULLISH":
        vote("BULL", 2.5, "bullish BOS")
    elif structure["bos"] == "BEARISH":
        vote("BEAR", 2.5, "bearish BOS")
    else:
        max_points += 2.5

    if structure["choch"] == "BULLISH":
        vote("BULL", 1.5, "bullish CHoCH")
    elif structure["choch"] == "BEARISH":
        vote("BEAR", 1.5, "bearish CHoCH")
    else:
        max_points += 1.5

    # Liquidity sweep
    if sweep["type"] == "BULLISH_SWEEP":
        vote("BULL", 1.5, "bullish liquidity sweep")
    elif sweep["type"] == "BEARISH_SWEEP":
        vote("BEAR", 1.5, "bearish liquidity sweep")
    else:
        max_points += 1.5

    # Displacement
    if displacement["direction"] == "BULLISH":
        vote("BULL", 1.5, "bullish displacement")
    elif displacement["direction"] == "BEARISH":
        vote("BEAR", 1.5, "bearish displacement")
    else:
        max_points += 1.5

    # FVG context — supporting evidence only
    recent_gaps = gaps[-8:]
    has_bull_fvg = any(g["direction"] == "BULL" for g in recent_gaps)
    has_bear_fvg = any(g["direction"] == "BEAR" for g in recent_gaps)
    if has_bull_fvg and not has_bear_fvg:
        vote("BULL", 1.0, "bullish FVG context")
    elif has_bear_fvg and not has_bull_fvg:
        vote("BEAR", 1.0, "bearish FVG context")
    else:
        max_points += 1.0

    # Order-block proxy — intentionally low weight
    recent_obs = obs[-6:]
    has_bull_ob = any(o["direction"] == "BULL" for o in recent_obs)
    has_bear_ob = any(o["direction"] == "BEAR" for o in recent_obs)
    if has_bull_ob and not has_bear_ob:
        vote("BULL", 0.75, "bullish order-block proxy")
    elif has_bear_ob and not has_bull_ob:
        vote("BEAR", 0.75, "bearish order-block proxy")
    else:
        max_points += 0.75

    # ADX / volume / volatility are confidence modifiers, not direction generators
    directional_side = "BULL" if bull > bear else "BEAR" if bear > bull else None

    if np.isfinite(adx):
        max_points += 1.25
        if adx >= 25 and directional_side:
            if directional_side == "BULL":
                bull += 1.25
            else:
                bear += 1.25
            reasons.append("ADX confirms trend strength")

    max_points += 1.0
    if vol_exp >= 1.5 and directional_side:
        if directional_side == "BULL":
            bull += 1.0
        else:
            bear += 1.0
        reasons.append("volume expansion confirms move")

    # Expansion helps; compression penalizes directional confidence.
    max_points += 1.0
    if regime == "EXPANSION" and directional_side:
        if directional_side == "BULL":
            bull += 1.0
        else:
            bear += 1.0
        reasons.append("volatility expansion")
    elif regime == "COMPRESSION":
        if directional_side == "BULL":
            bull = max(0.0, bull - 1.0)
        elif directional_side == "BEAR":
            bear = max(0.0, bear - 1.0)
        reasons.append("volatility compression penalty")

    score = bull - bear
    total_directional = bull + bear
    dominant = max(bull, bear)

    confluence_index = 0.0 if max_points <= 0 else 100.0 * dominant / max_points
    dominance = 0.0 if total_directional <= 0 else 100.0 * abs(bull - bear) / total_directional

    # Strict raw gate. Final gate below is even stricter.
    if score >= 4.0 and confluence_index >= 58 and dominance >= 25:
        raw = "LONG"
    elif score <= -4.0 and confluence_index >= 58 and dominance >= 25:
        raw = "SHORT"
    else:
        raw = "WAIT"

    return {
        "raw": raw,
        "score": score,
        "bull": bull,
        "bear": bear,
        "confluence_index": confluence_index,
        "dominance": dominance,
        "reasons": reasons,
        "atr": a,
        "rsi": rv,
        "adx": adx,
        "regime": regime,
        "regime_ratio": vol_ratio,
        "bos": structure["bos"],
        "choch": structure["choch"],
        "sweep": sweep["type"],
        "displacement": displacement["direction"],
        "volume_ratio": vol_exp,
        "vwap": vwap,
    }


# ============================================================
# TARGET / STOP CANDIDATE ENGINE
# ============================================================

def add_candidate(candidates, price, label, weight, entry, direction):
    if price is None or not np.isfinite(price):
        return
    if direction == "LONG" and price <= entry:
        return
    if direction == "SHORT" and price >= entry:
        return
    candidates.append({
        "price": float(price),
        "label": label,
        "weight": float(weight),
        "distance": abs(float(price) - float(entry)),
    })


def add_stop_candidate(candidates, price, label, entry, direction):
    if price is None or not np.isfinite(price):
        return
    if direction == "LONG" and price >= entry:
        return
    if direction == "SHORT" and price <= entry:
        return
    candidates.append({
        "price": float(price),
        "label": label,
        "distance": abs(float(price) - float(entry)),
    })


def choose_target_and_stop(df, direction, timeframe, tick_size, orb=None):
    if direction not in ("LONG", "SHORT") or df is None or df.empty:
        return None

    entry = float(df["Close"].iloc[-1])
    a = calculate_atr(df)
    if not np.isfinite(a) or a <= 0:
        return None

    highs, lows = swing_levels(df)
    gaps = recent_fvgs(df)
    obs = recent_order_blocks(df)
    levels = prior_day_week_levels(df)

    targets = []
    stops = []

    # ---------- swing highs/lows ----------
    for _, p in highs[-8:]:
        if direction == "LONG":
            add_candidate(targets, p, "swing high liquidity", 3.0, entry, direction)
        else:
            add_stop_candidate(stops, p, "swing-high invalidation", entry, direction)

    for _, p in lows[-8:]:
        if direction == "SHORT":
            add_candidate(targets, p, "swing low liquidity", 3.0, entry, direction)
        else:
            add_stop_candidate(stops, p, "swing-low invalidation", entry, direction)

    # ---------- PDH / PDL / PWH / PWL / session ----------
    if direction == "LONG":
        add_candidate(targets, levels.get("PDH"), "prior-day high", 4.0, entry, direction)
        add_candidate(targets, levels.get("PWH"), "prior-week high", 4.0, entry, direction)
        add_candidate(targets, levels.get("SESSION_HIGH"), "session high", 2.0, entry, direction)
        add_stop_candidate(stops, levels.get("PDL"), "prior-day low invalidation", entry, direction)
        add_stop_candidate(stops, levels.get("SESSION_LOW"), "session-low invalidation", entry, direction)
    else:
        add_candidate(targets, levels.get("PDL"), "prior-day low", 4.0, entry, direction)
        add_candidate(targets, levels.get("PWL"), "prior-week low", 4.0, entry, direction)
        add_candidate(targets, levels.get("SESSION_LOW"), "session low", 2.0, entry, direction)
        add_stop_candidate(stops, levels.get("PDH"), "prior-day high invalidation", entry, direction)
        add_stop_candidate(stops, levels.get("SESSION_HIGH"), "session-high invalidation", entry, direction)

    # ---------- FVGs ----------
    for g in gaps[-15:]:
        if direction == "LONG" and g["low"] > entry:
            add_candidate(targets, g["low"], "FVG near edge", 2.0, entry, direction)
            add_candidate(targets, g["high"], "FVG far edge", 1.5, entry, direction)
        elif direction == "SHORT" and g["high"] < entry:
            add_candidate(targets, g["high"], "FVG near edge", 2.0, entry, direction)
            add_candidate(targets, g["low"], "FVG far edge", 1.5, entry, direction)

        if direction == "LONG" and g["high"] < entry:
            add_stop_candidate(stops, g["low"], "FVG invalidation", entry, direction)
        elif direction == "SHORT" and g["low"] > entry:
            add_stop_candidate(stops, g["high"], "FVG invalidation", entry, direction)

    # ---------- order blocks ----------
    for ob in obs[-8:]:
        if direction == "LONG":
            if ob["direction"] == "BEAR" and ob["low"] > entry:
                add_candidate(targets, ob["low"], "opposing order block", 1.0, entry, direction)
            if ob["direction"] == "BULL" and ob["high"] < entry:
                add_stop_candidate(stops, ob["low"], "bullish order-block invalidation", entry, direction)
        else:
            if ob["direction"] == "BULL" and ob["high"] < entry:
                add_candidate(targets, ob["high"], "opposing order block", 1.0, entry, direction)
            if ob["direction"] == "BEAR" and ob["low"] > entry:
                add_stop_candidate(stops, ob["high"], "bearish order-block invalidation", entry, direction)

    # ---------- ORB ----------
    if orb and timeframe in ("1m", "5m", "30m"):
        if direction == "LONG":
            if orb["low"] < entry < orb["high"]:
                add_candidate(targets, orb["high"], "ORB high", 4.0, entry, direction)
                add_stop_candidate(stops, orb["low"], "ORB low invalidation", entry, direction)
            elif entry > orb["high"]:
                add_stop_candidate(stops, orb["high"], "ORB breakout invalidation", entry, direction)
        else:
            if orb["low"] < entry < orb["high"]:
                add_candidate(targets, orb["low"], "ORB low", 4.0, entry, direction)
                add_stop_candidate(stops, orb["high"], "ORB high invalidation", entry, direction)
            elif entry < orb["low"]:
                add_stop_candidate(stops, orb["low"], "ORB breakout invalidation", entry, direction)

    # ---------- ATR extension fallback ----------
    if direction == "LONG":
        add_candidate(targets, entry + 1.5 * a, "1.5 ATR extension", 1.0, entry, direction)
    else:
        add_candidate(targets, entry - 1.5 * a, "1.5 ATR extension", 1.0, entry, direction)

    # Score targets: favor meaningful structure without choosing something absurdly far away.
    for t in targets:
        dist_atr = t["distance"] / a
        distance_penalty = max(0.0, dist_atr - 3.0) * 0.75
        t["rank"] = t["weight"] - distance_penalty

    targets = sorted(
        targets,
        key=lambda x: (-x["rank"], x["distance"])
    )

    if not targets:
        return None

    chosen_target = targets[0]

    # nearest valid structure stop, but not unrealistically tight
    min_stop_distance = 0.35 * a
    valid_stops = [s for s in stops if s["distance"] >= min_stop_distance]

    if valid_stops:
        chosen_stop = min(valid_stops, key=lambda x: x["distance"])
        raw_stop = chosen_stop["price"]
        stop_label = chosen_stop["label"]
    else:
        raw_stop = entry - a if direction == "LONG" else entry + a
        stop_label = "1 ATR fallback invalidation"

    # modest volatility buffer beyond structure
    buffer = 0.10 * a
    stop = raw_stop - buffer if direction == "LONG" else raw_stop + buffer

    entry = round_to_tick(entry, tick_size)
    target = round_to_tick(chosen_target["price"], tick_size)
    stop = round_to_tick(stop, tick_size)

    reward = abs(target - entry)
    risk = abs(entry - stop)
    rr = reward / risk if risk > 0 else 0.0
    atr_multiple = reward / a if a > 0 else 0.0

    return {
        "entry": entry,
        "target": target,
        "target_label": chosen_target["label"],
        "stop": stop,
        "stop_label": stop_label,
        "reward_points": reward,
        "risk_points": risk,
        "rr": rr,
        "atr_multiple": atr_multiple,
    }


# ============================================================
# FINAL QUALIFICATION FILTER
# ============================================================

def qualify_signal(
    raw_direction,
    plan,
    tick_size,
    tick_value,
    contracts,
    minimum_profit,
    minimum_rr,
    minimum_atr_multiple,
    confluence_index,
    dominance,
    minimum_confluence_index,
    minimum_dominance,
):
    """
    Final OP.exe gate:
    LONG/SHORT survives only when BOTH market direction and trade structure are strong.
    The confluence index is not a win probability.
    """
    if raw_direction not in ("LONG", "SHORT") or not plan:
        return {
            "signal": "WAIT",
            "reason": "Direction is not strong enough",
            "projected_profit": 0.0,
            "projected_risk": 0.0,
        }

    profit = pnl_between(
        plan["entry"], plan["target"],
        tick_size, tick_value, contracts
    )
    risk = pnl_between(
        plan["entry"], plan["stop"],
        tick_size, tick_value, contracts
    )

    failures = []

    if confluence_index < minimum_confluence_index:
        failures.append(f"confluence < {minimum_confluence_index:.0f}/100")

    if dominance < minimum_dominance:
        failures.append(f"directional dominance < {minimum_dominance:.0f}/100")

    if profit < minimum_profit:
        failures.append(f"projected TP < ${minimum_profit:,.0f}")

    if plan["rr"] < minimum_rr:
        failures.append(f"R:R < {minimum_rr:.2f}:1")

    if plan["atr_multiple"] < minimum_atr_multiple:
        failures.append(f"target < {minimum_atr_multiple:.2f} ATR")

    if failures:
        return {
            "signal": "WAIT",
            "reason": "; ".join(failures),
            "projected_profit": profit,
            "projected_risk": risk,
        }

    return {
        "signal": raw_direction,
        "reason": "STRICT CONFLUENCE PASSED",
        "projected_profit": profit,
        "projected_risk": risk,
    }


# ============================================================
# DATA LOADER
# ============================================================

def load_all_data(symbol, prefer_topstep=True):
    username, api_key = get_topstep_credentials()
    token = None
    contract = None
    error = None

    if prefer_topstep and username and api_key:
        token, error = px_login(username, api_key)
        if token:
            try:
                contract = px_find_contract(token, symbol, live=True)
            except Exception as e:
                error = f"Contract search failed: {e}"
                contract = None

    frames = {}

    if token and contract:
        mode = "TopstepX / ProjectX live bars"

        for tf in TIMEFRAMES:
            try:
                frames[tf] = px_retrieve_bars(
                    token, contract["id"], tf, live=True
                )
            except Exception as e:
                frames[tf] = pd.DataFrame()
                error = f"{tf} live-data error: {e}"

        root = clean_root(symbol)
        fallback = FALLBACK_SPECS.get(root, {})

        tick_size = float(
            contract.get("tickSize")
            or fallback.get("tick_size", 0.25)
        )
        tick_value = float(
            contract.get("tickValue")
            or fallback.get("tick_value", 1.0)
        )
        resolved = contract.get("name") or symbol

    else:
        mode = "Yahoo fallback — proxy/delayed"

        for tf in TIMEFRAMES:
            try:
                frames[tf] = yahoo_bars(symbol, tf)
            except Exception:
                frames[tf] = pd.DataFrame()

        root = clean_root(symbol)
        spec = FALLBACK_SPECS.get(root, {
            "tick_size": 0.01,
            "tick_value": 1.0,
            "yf": symbol,
        })
        tick_size = float(spec["tick_size"])
        tick_value = float(spec["tick_value"])
        resolved = spec.get("yf", symbol)

    return {
        "frames": frames,
        "mode": mode,
        "error": error,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "resolved": resolved,
        "contract": contract,
    }



# ============================================================
# CROSS-TIMEFRAME CONFLUENCE
# ============================================================

TF_ORDER = ["1m", "5m", "30m", "4h", "1d", "1w"]

def higher_timeframe_alignment(raw_results, tf):
    """
    Returns a small supporting adjustment based on adjacent/higher timeframes.
    This does NOT override a timeframe's own structure.
    """
    if tf not in TF_ORDER:
        return 0.0, "none"

    i = TF_ORDER.index(tf)
    direction = raw_results.get(tf, {}).get("raw", "WAIT")
    if direction not in ("LONG", "SHORT"):
        return 0.0, "none"

    checks = []
    for higher in TF_ORDER[i+1:i+3]:
        hdir = raw_results.get(higher, {}).get("raw", "WAIT")
        if hdir in ("LONG", "SHORT"):
            checks.append(hdir)

    if not checks:
        return 0.0, "no higher-timeframe confirmation"

    same = sum(1 for x in checks if x == direction)
    opposite = sum(1 for x in checks if x != direction)

    if same >= 2:
        return 7.5, "strong higher-timeframe alignment"
    if same == 1 and opposite == 0:
        return 4.0, "higher-timeframe alignment"
    if opposite >= 1:
        return -7.5, "higher-timeframe conflict"
    return 0.0, "mixed higher-timeframe evidence"


# ============================================================
# USER INTERFACE
# ============================================================

st.title("OP.exe")
st.caption(
    "PRIMARY GOAL: independent LONG / SHORT / WAIT for each timeframe using live market data • Structure • BOS • Sweeps • FVGs • ORB • TP/SL"
)

with st.form("op_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2.0, 0.9, 1.15, 1.35])

    with c1:
        symbol = st.text_input(
            "Ticker / Contract",
            value=st.session_state.get("symbol", "MNQ"),
            placeholder="MNQ, MES, MGC, MCL..."
        )

    with c2:
        contracts = st.number_input(
            "Contracts",
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )

    with c3:
        minimum_profit = st.number_input(
            "Min projected profit",
            min_value=0,
            max_value=100000,
            value=200,
            step=50
        )

    with c4:
        orb_mode = st.selectbox(
            "ORB",
            ["09:00–09:30 ET", "09:30–10:00 ET"],
            index=0
        )

    run_button = st.form_submit_button(
        "Run OP.exe",
        use_container_width=True
    )

if run_button:
    st.session_state["symbol"] = symbol.upper().strip()

symbol = st.session_state.get(
    "symbol",
    symbol.upper().strip() or "MNQ"
)

with st.expander("Advanced filters", expanded=False):
    a1, a2, a3, a4 = st.columns(4)

    with a1:
        minimum_rr = st.number_input(
            "Minimum reward:risk",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.25
        )

    with a2:
        minimum_atr_multiple = st.number_input(
            "Minimum TP distance (ATR)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25
        )

    with a3:
        auto_refresh = st.toggle("Auto refresh", value=False)

    with a4:
        refresh_seconds = st.number_input(
            "Refresh seconds",
            min_value=15,
            max_value=300,
            value=30,
            step=15
        )

    b1, b2 = st.columns(2)
    with b1:
        minimum_confluence_index = st.number_input(
            "Minimum confluence index (0-100)",
            min_value=50.0,
            max_value=100.0,
            value=72.0,
            step=1.0,
            help="This is an OP.exe evidence-agreement score for THIS timeframe only, NOT a probability of winning."
        )
    with b2:
        minimum_dominance = st.number_input(
            "Minimum directional dominance (0-100)",
            min_value=20.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
            help="How one-sided the bullish vs bearish evidence must be within THIS timeframe."
        )

    prefer_topstep = st.toggle(
        "Use TopstepX live market data when API credentials are configured",
        value=True
    )

if auto_refresh:
    st_autorefresh(
        interval=int(refresh_seconds * 1000),
        key="op_refresh"
    )

with st.spinner("Reading market data and calculating structure..."):
    bundle = load_all_data(symbol, prefer_topstep=prefer_topstep)

frames = bundle["frames"]
tick_size = bundle["tick_size"]
tick_value = bundle["tick_value"]
mode = bundle["mode"]

# choose freshest available price
last_price = np.nan
for tf in ["1m", "5m", "30m", "4h", "1d", "1w"]:
    df = frames.get(tf, pd.DataFrame())
    if df is not None and not df.empty:
        last_price = float(df["Close"].iloc[-1])
        break

m1, m2, m3, m4 = st.columns(4)
m1.metric("Symbol", bundle["resolved"])
m2.metric("Price", f"{last_price:,.2f}" if np.isfinite(last_price) else "No data")
m3.metric("Market data", "LIVE" if mode.startswith("TopstepX") else "FALLBACK")
m4.metric(f"{int(contracts)}-contract threshold", f"${minimum_profit:,.0f}")

is_live_feed = mode.startswith("TopstepX")

if is_live_feed:
    st.success("LIVE MODE: TopstepX / ProjectX live bars connected.")
else:
    st.warning(
        "FALLBACK MODE: Yahoo Finance data is being used. It may be delayed/proxy data and is NOT treated as true real-time."
    )

if bundle["error"]:
    st.caption(bundle["error"])

orb = latest_orb(
    frames.get("1m", pd.DataFrame()),
    orb_mode
)

if orb and np.isfinite(last_price):
    if orb["low"] <= last_price <= orb["high"]:
        orb_state = "INSIDE ORB"
    elif last_price > orb["high"]:
        orb_state = "ABOVE ORB"
    else:
        orb_state = "BELOW ORB"

    st.caption(
        f'{orb["mode"]} | {orb["date"]} | '
        f'Low {orb["low"]:,.2f} • High {orb["high"]:,.2f} • '
        f'Mid {orb["mid"]:,.2f} • {orb_state}'
    )

calculator_tab, diagnostics_tab = st.tabs(
    ["Calculator", "Structure & diagnostics"]
)

results = {}

# ============================================================
# PAGE 1 — CLEAN CALCULATOR
# ============================================================

with calculator_tab:

    # Every timeframe is evaluated on its own.
    # A 1m signal does NOT need 5m/30m confirmation.
    # A 5m signal does NOT need 1m/30m confirmation, etc.
    raw_evidence = {
        tf: evidence_engine(frames.get(tf, pd.DataFrame()), tf)
        for tf in TIMEFRAMES
    }

    for tf in TIMEFRAMES:
        df = frames.get(tf, pd.DataFrame())
        evidence = dict(raw_evidence[tf])
        evidence["htf_note"] = "independent timeframe"

        plan = (
            choose_target_and_stop(
                df,
                evidence["raw"],
                tf,
                tick_size,
                orb=orb,
            )
            if not df.empty
            else None
        )

        qualified = qualify_signal(
            evidence["raw"],
            plan,
            tick_size,
            tick_value,
            int(contracts),
            float(minimum_profit),
            float(minimum_rr),
            float(minimum_atr_multiple),
            float(evidence.get("confluence_index", 0.0)),
            float(evidence.get("dominance", 0.0)),
            float(minimum_confluence_index),
            float(minimum_dominance),
        )

        result = {
            **evidence,
            **qualified,
            "plan": plan,
        }
        results[tf] = result

        signal = result["signal"]

        if signal == "LONG":
            card_class = "op-long"
        elif signal == "SHORT":
            card_class = "op-short"
        else:
            card_class = "op-wait"

        if plan:
            entry_text = f'{plan["entry"]:,.2f}'
            tp_text = f'{plan["target"]:,.2f}'
            sl_text = f'{plan["stop"]:,.2f}'
            rr_text = f'{plan["rr"]:.2f}:1'
            move_text = f'{plan["atr_multiple"]:.2f}x ATR'
            tp_reason = plan["target_label"]
            sl_reason = plan["stop_label"]
        else:
            entry_text = tp_text = sl_text = rr_text = move_text = "—"
            tp_reason = sl_reason = "—"

        st.markdown(
            f"""
            <div class="op-card {card_class}">
              <div style="display:flex;justify-content:space-between;gap:10px;align-items:center">
                <div>
                  <b style="font-size:1.08rem">{tf}</b>
                  &nbsp;&nbsp;<b>{signal}</b>
                </div>
                <div class="muted">{result["reason"]}</div>
              </div>

              <div style="margin-top:8px">
                Entry <b>{entry_text}</b> &nbsp;•&nbsp;
                TP <b>{tp_text}</b> &nbsp;•&nbsp;
                SL <b>{sl_text}</b> &nbsp;•&nbsp;
                R:R <b>{rr_text}</b> &nbsp;•&nbsp;
                Move <b>{move_text}</b> &nbsp;•&nbsp;
                Confluence <b>{result.get("confluence_index", 0.0):.0f}/100</b>
              </div>

              <div style="margin-top:5px">
                Projected TP ({int(contracts)} contracts):
                <b>${result["projected_profit"]:,.0f}</b>
                &nbsp;•&nbsp;
                Projected SL:
                <b>-${result["projected_risk"]:,.0f}</b>
              </div>

              <div class="tiny" style="margin-top:5px">
                TP basis: {tp_reason} &nbsp;•&nbsp;
                SL basis: {sl_reason} &nbsp;•&nbsp;
                Mode: independent timeframe
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "INDEPENDENT TIMEFRAME MODE: each row stands alone. "
        "1m, 5m and 30m do not have to agree for a timeframe to output LONG or SHORT."
    )

# ============================================================
# PAGE 2 — STRUCTURE / DIAGNOSTICS
# ============================================================

with diagnostics_tab:

    st.subheader("Structure map")

    rows = []

    for tf, r in results.items():
        p = r.get("plan")

        rows.append({
            "TF": tf,
            "Raw": r.get("raw"),
            "Final": r.get("signal"),
            "Score": r.get("score"),
            "Bull pts": r.get("bull"),
            "Bear pts": r.get("bear"),
            "Confluence /100": round(r.get("confluence_index", 0.0), 1),
            "Dominance /100": round(r.get("dominance", 0.0), 1),
            "Mode": "Independent timeframe",
            "BOS": r.get("bos"),
            "CHoCH": r.get("choch"),
            "Sweep": r.get("sweep"),
            "Displacement": r.get("displacement"),
            "Regime": r.get("regime"),
            "ADX": round(r.get("adx"), 1) if np.isfinite(r.get("adx", np.nan)) else None,
            "RSI": round(r.get("rsi"), 1) if np.isfinite(r.get("rsi", np.nan)) else None,
            "Volume x": round(r.get("volume_ratio", 1.0), 2),
            "Entry": p.get("entry") if p else None,
            "TP": p.get("target") if p else None,
            "SL": p.get("stop") if p else None,
            "R:R": round(p.get("rr"), 2) if p else None,
            "Projected TP $": round(r.get("projected_profit", 0.0), 2),
            "Projected SL $": round(r.get("projected_risk", 0.0), 2),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("What is weighted most heavily")

    st.write(
        "OP.exe gives more weight to trend, break of structure, price location, "
        "risk/reward and volatility. Liquidity sweeps, FVGs, displacement and "
        "order-block proxies are supporting evidence rather than stand-alone guarantees."
    )

    st.subheader("Live data setup")

    st.code(
        '# Streamlit Cloud → App settings → Secrets\n'
        'TOPSTEP_USERNAME = "your username"\n'
        'TOPSTEP_API_KEY = "your API key"',
        language="toml",
    )

    st.caption(
        "Never put your API key directly into app.py or commit it to GitHub."
    )

    st.subheader("Important limits")

    st.write(
        "No combination of technical indicators, market-structure labels, take-profits "
        "or stop-losses can make a trade certain. Order blocks, FVGs, liquidity sweeps "
        "and similar concepts are implemented here as explicit, testable rules rather "
        "than assumed facts. A stop is an invalidation level, not a guaranteed fill price."
    )

st.divider()
st.caption(
    "OP.exe confluence scores measure agreement among its rules; they are not win probabilities or guarantees. Backtest and paper-test before live use."
)

