"""
BTC/USDT — статистическая проверка торговых гипотез.
Live-демо к проекту btc-market-analysis: превращает ноутбук в интерактивный дашборд.

Данные: дневные свечи BTC/USDT с Binance (2017–наши дни).
По умолчанию грузятся из btc_daily.csv (надёжно на облаке); кнопкой можно обновить live через ccxt.

Запуск локально:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import argrelextrema

st.set_page_config(page_title="BTC/USDT · бэктест паттернов", page_icon="📈", layout="wide")

BTC_ORANGE = "#F7931A"
CSV_PATH = os.path.join(os.path.dirname(__file__), "btc_daily.csv")


# ─────────────────────────────── Данные ───────────────────────────────
@st.cache_data(show_spinner=False)
def load_from_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
    return df


@st.cache_data(show_spinner="Тяну свежие свечи с Binance…")
def load_live() -> pd.DataFrame:
    import ccxt

    ex = ccxt.binance()
    all_data, limit = [], 1000
    since = ex.parse8601("2017-08-01T00:00:00Z")
    while True:
        ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1d", since=since, limit=limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break
        time.sleep(0.3)
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


# ─────────────────────────── Индикаторы / логика ───────────────────────────
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["daily_return"] = df["close"].pct_change() * 100
    df["rsi"] = calculate_rsi(df["close"])
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["day_of_week"] = df.index.day_name()
    df["month"] = df.index.month_name()
    return df


def find_levels(df: pd.DataFrame, order: int):
    lo = argrelextrema(df["close"].values, np.less, order=order)[0]
    hi = argrelextrema(df["close"].values, np.greater, order=order)[0]
    return df["close"].iloc[lo], df["close"].iloc[hi]


def backtest_support(df, support_levels, threshold, fwd_days):
    rows = []
    for s_date, s_price in support_levels.items():
        fut = df[df.index > s_date]
        for date, row in fut.iterrows():
            idx = fut.index.get_loc(date)
            if abs(row["close"] - s_price) / s_price <= threshold and idx + fwd_days < len(fut):
                p0 = row["close"]
                pN = fut["close"].iloc[idx + fwd_days]
                rows.append({"return": (pN - p0) / p0 * 100, "bounce": pN > p0})
    return pd.DataFrame(rows)


def breakout_volume(df, resistance_levels, threshold=0.01, fwd_days=3):
    rows = []
    for r_date, r_price in resistance_levels.items():
        fut = df[df.index > r_date]
        for date, row in fut.iterrows():
            idx = fut.index.get_loc(date)
            if row["close"] > r_price * (1 + threshold) and idx + fwd_days < len(fut):
                p3 = fut["close"].iloc[idx + fwd_days]
                all_idx = df.index.get_loc(date)
                avg_vol_20 = df["volume"].iloc[max(0, all_idx - 20):all_idx].mean()
                rows.append({
                    "volume_ratio": row["volume"] / avg_vol_20 if avg_vol_20 else np.nan,
                    "false_breakout": p3 < r_price,
                })
                break
    return pd.DataFrame(rows).dropna()


# ─────────────────────────────── Sidebar ───────────────────────────────
st.sidebar.title("⚙️ Параметры")
source = st.sidebar.radio("Источник данных", ["CSV (быстро)", "Live с Binance"], index=0)
raw = load_live() if source.startswith("Live") else load_from_csv()

max_years = round((raw.index.max() - raw.index.min()).days / 365, 1)
years = st.sidebar.slider("Глубина истории, лет", 1.0, float(max_years), float(max_years), 0.5)
cutoff = raw.index.max() - pd.Timedelta(days=int(years * 365))
df = enrich(raw[raw.index >= cutoff])

order = st.sidebar.slider("Окно для уровней (order)", 5, 20, 10,
                          help="Локальный экстремум сильнее соседей в окне ±order дней")
threshold = st.sidebar.slider("Порог касания поддержки, %", 1.0, 5.0, 2.0, 0.5) / 100
fwd_days = st.sidebar.slider("Горизонт отскока, дней", 1, 14, 3)
st.sidebar.caption("Двигай ползунки — все гипотезы пересчитываются вживую.")

support, resistance = find_levels(df, order)

# ─────────────────────────────── Header ───────────────────────────────
st.title("📈 BTC/USDT — статистическая проверка торговых гипотез")
st.caption(f"Бэктест на дневных свечах Binance. Каждая гипотеза проверена t-тестом. "
           f"Автор: [github.com/sanykiv](https://github.com/sanykiv/btc-market-analysis)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Свечей в выборке", f"{len(df):,}")
c2.metric("Период", f"{df.index.min():%Y-%m} → {df.index.max():%Y-%m}")
c3.metric("Последняя цена", f"${df['close'].iloc[-1]:,.0f}")
c4.metric("Уровней S / R", f"{len(support)} / {len(resistance)}")

tab_price, tab_bt, tab_seas, tab_vol = st.tabs(
    ["Цена · RSI · уровни", "Бэктест отскока", "Сезонность", "Объём на пробоях"]
)

# ── Таб 1: цена / RSI / уровни ──
with tab_price:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot(df.index, df["close"], color=BTC_ORANGE, lw=1.3, label="BTC/USDT")
    ax1.plot(df.index, df["sma_50"], color="#9C27B0", lw=1, label="SMA 50")
    ax1.plot(df.index, df["sma_200"], color="#F44336", lw=1.2, label="SMA 200")
    for _, p in support.items():
        ax1.axhline(p, color="green", lw=0.4, alpha=0.25)
    for _, p in resistance.items():
        ax1.axhline(p, color="red", lw=0.4, alpha=0.25)
    ax1.set_ylabel("Цена (USDT)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(df.index, df["rsi"], color="#9467bd", lw=1, label="RSI(14)")
    ax2.axhline(70, color="red", ls="--", alpha=0.6)
    ax2.axhline(30, color="green", ls="--", alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("Зелёные линии — уровни поддержки, красные — сопротивления "
                "(локальные экстремумы, `argrelextrema`).")

# ── Таб 2: бэктест отскока ──
with tab_bt:
    bt = backtest_support(df, support, threshold, fwd_days)
    if len(bt) < 3:
        st.warning("Слишком мало касаний на этой выборке — увеличь глубину истории или порог касания.")
    else:
        t_stat, p_val = stats.ttest_1samp(bt["return"], 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Касаний поддержки", f"{len(bt)}")
        m2.metric(f"Winrate ({fwd_days}д)", f"{bt['bounce'].mean()*100:.1f}%")
        m3.metric("Средняя доходность", f"{bt['return'].mean():+.2f}%")
        m4.metric("p-value", f"{p_val:.3f}",
                  delta="значимо" if p_val < 0.05 else "не значимо",
                  delta_color="normal" if p_val < 0.05 else "inverse")
        fig2, ax = plt.subplots(figsize=(11, 4.5))
        ax.hist(bt["return"], bins=40, color="#4C72B0", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1.4, label="0%")
        ax.axvline(bt["return"].mean(), color="orange", ls="--", lw=1.4,
                   label=f"Среднее: {bt['return'].mean():+.2f}%")
        ax.set_title(f"Доходность через {fwd_days} дн. после касания поддержки "
                     f"(n={len(bt)}, p={p_val:.3f})")
        ax.set_xlabel("Доходность (%)")
        ax.set_ylabel("Случаев")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        st.info("💡 Гипотеза «цена отскакивает от поддержки». На короткой выборке часто "
                "незначима (p>0.05), на 9-летней — значима. Размер выборки решает.")

# ── Таб 3: сезонность ──
with tab_seas:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    mode = st.radio("Разрез", ["По дням недели", "По месяцам"], horizontal=True)
    keys = day_order if mode.startswith("По дн") else month_order
    col = "day_of_week" if mode.startswith("По дн") else "month"

    means, colors, labels = [], [], []
    for k in keys:
        r = df[df[col] == k]["daily_return"].dropna()
        if len(r) < 2:
            means.append(0); colors.append("#cccccc"); labels.append(k[:3]); continue
        _, p = stats.ttest_1samp(r, 0)
        mean = r.mean()
        means.append(mean)
        if p < 0.05:
            colors.append("#2ecc71" if mean > 0 else "#e74c3c")
        else:
            colors.append("#a8d8a8" if mean > 0 else "#f4a9a8")
        labels.append(k[:3])
    fig3, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(labels, means, color=colors, alpha=0.9, edgecolor="white")
    ax.bar_label(bars, fmt=lambda x: f"{x:.2f}", fontsize=8, padding=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Средняя дневная доходность (%)")
    ax.set_title(f"{mode} · яркий цвет = статистически значимо (p<0.05)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Ярко-зелёный/красный — значимый эффект; бледный — не значим (шум).")

# ── Таб 4: объём на пробоях ──
with tab_vol:
    bv = breakout_volume(df, resistance, threshold=0.01, fwd_days=3)
    if len(bv) < 4 or bv["false_breakout"].nunique() < 2:
        st.warning("Мало пробоев на этой выборке — увеличь глубину истории.")
    else:
        true_b = bv[~bv["false_breakout"]]["volume_ratio"]
        false_b = bv[bv["false_breakout"]]["volume_ratio"]
        t_stat, p_val = stats.ttest_ind(true_b, false_b)
        m1, m2, m3 = st.columns(3)
        m1.metric("Истинных пробоев", f"{len(true_b)} · {true_b.mean():.2f}×")
        m2.metric("Ложных пробоев", f"{len(false_b)} · {false_b.mean():.2f}×")
        m3.metric("p-value", f"{p_val:.3f}",
                  delta="значимо" if p_val < 0.05 else "не значимо",
                  delta_color="normal" if p_val < 0.05 else "inverse")
        fig4, ax = plt.subplots(figsize=(9, 4.5))
        bp = ax.boxplot([true_b.values, false_b.values],
                        tick_labels=["Истинный", "Ложный"], patch_artist=True,
                        medianprops=dict(color="red", lw=2))
        bp["boxes"][0].set_facecolor("#4C72B0"); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#DD8452"); bp["boxes"][1].set_alpha(0.6)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.7, label="Норма (1×)")
        ax.set_ylabel("Volume ratio (к SMA20)")
        ax.set_title(f"Объём при пробоях (n={len(bv)}, p={p_val:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        st.info("💡 Истинные пробои идут на повышенном объёме (~1.9× от нормы) против ложных (~1.3×). "
                "Объём — фильтр ложных пробоев.")

st.divider()
st.caption("Проект и код: github.com/sanykiv/btc-market-analysis · "
           "данные Binance через ccxt · не является финансовой рекомендацией.")
