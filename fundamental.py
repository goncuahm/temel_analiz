"""
BİST Temettü Hisseleri — Temel Analiz & DDM Değerleme Uygulaması
Streamlit app with fundamental analysis, DDM, DuPont, and peer comparison
for Turkish dividend-paying stocks.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BİST Temel Analiz",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark financial terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #0f1e30 !important;
    border: 1px solid #1e4a7a !important;
    color: #e0f0ff !important;
}

/* Main background */
.main .block-container { background: #070d18; padding-top: 1.5rem; }
.stApp { background: #070d18; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1f35 0%, #091528 100%);
    border: 1px solid #1a3a5c;
    border-left: 3px solid #00aaff;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    color: #6a9ec0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #e8f4ff;
}
.metric-card .delta {
    font-size: 0.75rem;
    margin-top: 2px;
}
.positive { color: #00e676; }
.negative { color: #ff5252; }
.neutral  { color: #80cbc4; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #00aaff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* DDM result box */
.ddm-result {
    background: linear-gradient(135deg, #071a0f 0%, #0a2015 100%);
    border: 1px solid #1a5c3a;
    border-left: 4px solid #00e676;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
}
.ddm-result.overvalued {
    background: linear-gradient(135deg, #1a0707 0%, #200a0a 100%);
    border-color: #5c1a1a;
    border-left-color: #ff5252;
}
.ddm-result .ticker-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.80rem;
    color: #6a9ec0;
    letter-spacing: 0.1em;
}
.ddm-result .fair-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #00e676;
}
.ddm-result.overvalued .fair-value { color: #ff5252; }

/* Dataframe styling */
.stDataFrame { background: #0a1525 !important; }

/* Title */
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8f4ff;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    color: #4a7a9b;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    background: #0d2a45;
    border: 1px solid #1a4a70;
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #5ab4e0;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-title">📊 BİST Temel Analiz Merkezi</div>
<div class="app-subtitle">
    <span class="badge">DDM</span>
    <span class="badge">DuPont</span>
    <span class="badge">Göreceli Değerleme</span>
    <span class="badge">Finansal Rasyolar</span>
    Türk temettü hisseleri için kapsamlı temel analiz
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametreler")
    st.markdown("---")

    DEFAULT_TICKERS = ["BASGZ.IS", "ENJSA.IS", "TUPRS.IS", "AYEN.IS", "AKGRT.IS", "AYGAZ.IS"]

    ticker_raw = st.text_area(
        "Hisse Kodları (virgülle ayırın)",
        value=", ".join(DEFAULT_TICKERS),
        height=130,
        help="Yahoo Finance formatı: TUPRS.IS, ENJSA.IS vb."
    )
    tickers = [t.strip().upper() for t in ticker_raw.replace("\n", ",").split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### 📐 DDM Parametreleri")
    rf = st.slider("Risksiz Oran — Rf (%)", 20.0, 35.0, 28.0, 0.5,
                   help="10 yıllık TL tahvil getirisi") / 100
    erp = st.slider("Piyasa Risk Primi — ERP (%)", 5.0, 15.0, 9.0, 0.5,
                    help="Damodaran Türkiye ERP baz değeri") / 100
    g_default = st.slider("Varsayılan Büyüme Oranı — g (%)", 3.0, 15.0, 7.0, 0.5,
                           help="DDM'de kullanılacak uzun vadeli temettü büyümesi") / 100

    st.markdown("---")
    st.markdown("### 🔄 Veri")
    if st.button("🔄 Verileri Yenile", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Not")
    st.caption("Temettü verimi, Yahoo'nun `.IS` hisseleri için hatalı döndürdüğü `trailingAnnualDividendYield` yerine **son 12 ay temettü toplamı / güncel fiyat** formülüyle hesaplanmaktadır.")

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all(tickers_list):
    rows = []
    hist_data = {}
    div_data = {}

    for t in tickers_list:
        try:
            stk = yf.Ticker(t)
            info = stk.info

            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("previousClose"))

            # ── Price history (2 years) ──
            hist = stk.history(period="2y")
            hist_data[t] = hist

            # ── RSI (14-day) ──
            rsi = None
            if len(hist) >= 20:
                d = hist["Close"].diff()
                gain = d.clip(lower=0).rolling(14).mean()
                loss = (-d.clip(upper=0)).rolling(14).mean()
                rs = gain / loss
                rsi = round((100 - 100 / (1 + rs)).iloc[-1], 1)

            # ── Dividend history ──
            divs = stk.dividends
            div_data[t] = divs

            # Last 3 years of annual dividends
            annual_divs = {}
            trailing_12m_div = 0.0
            if not divs.empty:
                divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
                for yr in [2022, 2023, 2024]:
                    yr_divs = divs[divs.index.year == yr]
                    annual_divs[yr] = round(yr_divs.sum(), 4) if not yr_divs.empty else None
                # Trailing 12-month dividends: sum all payments in last 365 days
                cutoff = divs.index[-1] - pd.Timedelta(days=365)
                trailing_12m_div = round(divs[divs.index >= cutoff].sum(), 4)

            # Calculate yield ourselves: trailing 12m dividends / current price
            # Yahoo's trailingAnnualDividendYield is unreliable for .IS stocks
            annual_div_rate = trailing_12m_div if trailing_12m_div > 0 else (info.get("trailingAnnualDividendRate") or 0)
            if annual_div_rate > 0 and price and price > 0:
                div_yield = round(annual_div_rate / price * 100, 2)
            else:
                # Last fallback: use Yahoo's field if available and non-zero
                yf_yield = (info.get("trailingAnnualDividendYield") or 0) * 100
                div_yield = round(yf_yield, 2)

            rows.append({
                "Ticker":        t,
                "Şirket":        (info.get("longName") or t)[:28],
                "Fiyat":         price,
                "RSI (14)":      rsi,
                "Temettü V. %":  div_yield,           # calculated: trailing_12m / price
                "Yıllık Tem.":   round(float(annual_div_rate), 3),
                "Tem. 2022":     annual_divs.get(2022),
                "Tem. 2023":     annual_divs.get(2023),
                "Tem. 2024":     annual_divs.get(2024),
                "F/K":           info.get("trailingPE"),
                "FD/FAVÖK":      info.get("enterpriseToEbitda"),
                "PD/DD":         info.get("priceToBook"),
                "ROE %":         round((info.get("returnOnEquity") or 0) * 100, 2),
                "ROA %":         round((info.get("returnOnAssets") or 0) * 100, 2),
                "Net Kar Marjı %": round((info.get("profitMargins") or 0) * 100, 2),
                "Brüt Marj %":   round((info.get("grossMargins") or 0) * 100, 2),
                "FAVÖK Marjı %": round((info.get("ebitdaMargins") or 0) * 100, 2),
                "Borç/Özkaynak": round(info.get("debtToEquity") or 0, 2),
                "Cari Oran":     round(info.get("currentRatio") or 0, 2),
                "Beta":          round(info.get("beta") or 1.0, 2),
                "Piy. Değ. (Mn TL)": round((info.get("marketCap") or 0) / 1e6, 0),
                "Sektör":        info.get("sector", "—"),
            })
        except Exception as e:
            rows.append({
                "Ticker": t, "Şirket": t,
                "Fiyat": None, "RSI (14)": None, "Temettü V. %": 0,
                "Yıllık Tem.": 0, "Tem. 2022": None, "Tem. 2023": None, "Tem. 2024": None,
                "F/K": None, "FD/FAVÖK": None, "PD/DD": None,
                "ROE %": None, "ROA %": None, "Net Kar Marjı %": None,
                "Brüt Marj %": None, "FAVÖK Marjı %": None,
                "Borç/Özkaynak": None, "Cari Oran": None, "Beta": None,
                "Piy. Değ. (Mn TL)": None, "Sektör": "—",
            })

    df = pd.DataFrame(rows).set_index("Ticker")
    return df, hist_data, div_data


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if not tickers:
    st.warning("Lütfen en az bir hisse kodu girin.")
    st.stop()

with st.spinner("📡 Veriler yükleniyor..."):
    df, hist_data, div_data = fetch_all(tuple(tickers))

valid = df[df["Fiyat"].notna()]
if valid.empty:
    st.error("Hiçbir hisse için veri alınamadı. Kodları kontrol edin.")
    st.stop()

# ─────────────────────────────────────────────
# DDM CALCULATION
# ─────────────────────────────────────────────
def calc_ddm(row, rf, erp, g_default):
    """Gordon Growth DDM: P0 = D1 / (Ke - g)"""
    beta = row["Beta"] if pd.notna(row["Beta"]) and row["Beta"] > 0 else 1.0
    ke = rf + beta * erp

    # Use most recent available dividend
    d0 = None
    for yr in [2024, 2023, 2022]:
        col = f"Tem. {yr}"
        if pd.notna(row.get(col)) and row.get(col, 0) > 0:
            d0 = row[col]
            break
    if d0 is None:
        d0 = row["Yıllık Tem."] if row["Yıllık Tem."] > 0 else None

    if d0 is None or d0 == 0:
        return None, None, None, ke, g_default

    # Compute dividend growth from history if possible
    t2022 = row.get("Tem. 2022")
    t2024 = row.get("Tem. 2024")
    if pd.notna(t2022) and pd.notna(t2024) and t2022 > 0 and t2024 > 0:
        g_hist = (t2024 / t2022) ** 0.5 - 1  # 2-year CAGR
        # Cap g between 2% and 25% for sanity
        g = min(max(g_hist, 0.02), 0.25)
    else:
        g = g_default

    if ke <= g:  # model breaks down
        return None, d0, g, ke, ke

    d1 = d0 * (1 + g)
    p0 = d1 / (ke - g)
    return round(p0, 2), round(d0, 3), round(g * 100, 1), round(ke * 100, 1), round(g * 100, 1)


ddm_results = {}
for ticker, row in valid.iterrows():
    fair, d0, g_used, ke, _ = calc_ddm(row, rf, erp, g_default)
    price = row["Fiyat"]
    upside = ((fair / price - 1) * 100) if fair and price else None
    ddm_results[ticker] = {
        "D₀": d0,
        "g (%)": g_used,
        "Ke (%)": ke,
        "DDM Adil Değer": fair,
        "Mevcut Fiyat": price,
        "Potansiyel (%)": round(upside, 1) if upside else None,
    }

ddm_df = pd.DataFrame(ddm_results).T

# ─────────────────────────────────────────────
# SMART FUNDAMENTAL SCORE
# ─────────────────────────────────────────────
def normalize(series, ascending=True):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([50.0] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn) * 100
    return norm if ascending else 100 - norm

score_df = valid[["F/K", "PD/DD", "ROE %", "Net Kar Marjı %", "FAVÖK Marjı %", "Temettü V. %"]].copy()

scores = pd.DataFrame(index=score_df.index)
for col, asc, w in [
    ("F/K",            False, 0.20),  # lower P/E = better
    ("PD/DD",          False, 0.15),
    ("ROE %",          True,  0.20),
    ("Net Kar Marjı %",True,  0.15),
    ("FAVÖK Marjı %",  True,  0.15),
    ("Temettü V. %",   True,  0.15),
]:
    col_data = score_df[col].dropna()
    if len(col_data) > 1:
        scores[col] = normalize(score_df[col].fillna(score_df[col].median()), ascending=asc) * w
    else:
        scores[col] = 50 * w

valid["Temel Skor"] = scores.sum(axis=1).round(1)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Genel Bakış",
    "💰 DDM Değerleme",
    "📊 Finansal Rasyolar",
    "🔬 DuPont Analizi",
    "📈 Fiyat & Temettü Geçmişi",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Hisse Özet Kartları</div>', unsafe_allow_html=True)

    cols_per_row = 3
    ticker_list = valid.index.tolist()
    for i in range(0, len(ticker_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, ticker in enumerate(ticker_list[i:i+cols_per_row]):
            row = valid.loc[ticker]
            fiyat = row["Fiyat"]
            div_yield = row["Temettü V. %"]
            rsi = row["RSI (14)"]
            skor = row.get("Temel Skor", "—")

            ddm_val = ddm_df.loc[ticker, "DDM Adil Değer"] if ticker in ddm_df.index else None
            upside = ddm_df.loc[ticker, "Potansiyel (%)"] if ticker in ddm_df.index else None

            rsi_color = "#ff5252" if rsi and rsi > 70 else "#00e676" if rsi and rsi < 30 else "#80cbc4"
            up_color = "positive" if upside and upside > 0 else "negative"

            with cols[j]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600; color:#e8f4ff;">{ticker}</span>
                        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; background:#0d2a45; padding:2px 8px; border-radius:3px; color:#5ab4e0;">SKOR: {skor}</span>
                    </div>
                    <div style="font-size:0.80rem; color:#4a7a9b; margin-bottom:10px;">{row['Şirket']}</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                        <div>
                            <div class="label">Fiyat (TL)</div>
                            <div class="value">{f"{fiyat:,.2f}" if fiyat else "—"}</div>
                        </div>
                        <div>
                            <div class="label">Temettü V.</div>
                            <div class="value" style="color:#00e676;">{f"%{div_yield:.1f}" if div_yield else "—"}</div>
                        </div>
                        <div>
                            <div class="label">RSI (14)</div>
                            <div class="value" style="color:{rsi_color};">{f"{rsi}" if rsi else "—"}</div>
                        </div>
                        <div>
                            <div class="label">DDM Potansiyel</div>
                            <div class="value {up_color}">{f"%{upside:+.1f}" if upside else "—"}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Smart Score ranking chart ──
    st.markdown('<div class="section-header">Temel Skor Sıralaması</div>', unsafe_allow_html=True)

    scored = valid.dropna(subset=["Temel Skor"]).sort_values("Temel Skor", ascending=True)
    if not scored.empty:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.55 * len(scored))))
        fig.patch.set_facecolor("#070d18")
        ax.set_facecolor("#0a1525")

        bar_colors = ["#00e676" if s >= 70 else "#f39c12" if s >= 50 else "#ff5252"
                      for s in scored["Temel Skor"]]
        bars = ax.barh(scored.index, scored["Temel Skor"], color=bar_colors, height=0.55,
                       edgecolor="none")

        # Add value labels
        for bar, val in zip(bars, scored["Temel Skor"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", ha="left",
                    color="#e8f4ff", fontsize=11, fontweight="bold",
                    fontfamily="monospace")

        ax.set_xlim(0, 115)
        ax.set_xlabel("Temel Skor (0–100)", color="#4a7a9b", fontsize=10)
        ax.tick_params(colors="#8ab0c8", labelsize=10)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.1, color="#2a4a6a")
        ax.set_title("Temel Skor: Yeşil ≥70 Mükemmel • Sarı ≥50 Makul • Kırmızı <50 Zayıf",
                     color="#4a7a9b", fontsize=9, pad=10)

        legend = [
            mpatches.Patch(color="#00e676", label="Mükemmel (≥70)"),
            mpatches.Patch(color="#f39c12", label="Makul (50-70)"),
            mpatches.Patch(color="#ff5252", label="Zayıf (<50)"),
        ]
        ax.legend(handles=legend, loc="lower right", framealpha=0.15,
                  facecolor="#0a1525", edgecolor="#1a3a5c", labelcolor="#c8d8e8", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Peer comparison overview table ──
    st.markdown('<div class="section-header">Peer Karşılaştırma Tablosu</div>', unsafe_allow_html=True)
    overview_cols = ["Şirket", "Fiyat", "Temettü V. %", "Yıllık Tem.", "F/K", "PD/DD", "ROE %", "Temel Skor"]
    overview = valid[overview_cols].copy()
    overview["Fiyat"] = overview["Fiyat"].map(lambda x: f"{x:,.2f} TL" if pd.notna(x) else "—")
    overview["Yıllık Tem."] = overview["Yıllık Tem."].map(lambda x: f"{x:.3f} TL" if pd.notna(x) else "—")

    def color_score(val):
        if pd.isna(val): return ""
        if val >= 70: return "background-color: #0a2a12; color: #00e676"
        if val >= 50: return "background-color: #2a1e06; color: #f39c12"
        return "background-color: #2a0a0a; color: #ff5252"

    styled = overview.style.applymap(color_score, subset=["Temel Skor"]) \
        .format({"Temettü V. %": "{:.2f}%", "F/K": "{:.1f}", "PD/DD": "{:.2f}", "ROE %": "{:.1f}%"},
                na_rep="—") \
        .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "0.85rem"})
    st.dataframe(styled, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — DDM VALUATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Gordon Growth Model — DDM Değerleme</div>', unsafe_allow_html=True)

    # Formula display
    st.markdown("""
    <div style="background:#0d1f35; border:1px solid #1a3a5c; border-radius:6px; padding:14px 20px; margin-bottom:20px; font-family:'IBM Plex Mono',monospace;">
        <span style="color:#4a7a9b; font-size:0.75rem;">FORMÜL</span><br>
        <span style="color:#e8f4ff; font-size:1.1rem;">P₀ = D₁ / (Ke − g) &nbsp;|&nbsp; D₁ = D₀ × (1+g) &nbsp;|&nbsp; Ke = Rf + β × ERP</span><br>
        <span style="color:#4a7a9b; font-size:0.75rem; margin-top:6px; display:block;">
            Rf = {:.1f}% &nbsp;|&nbsp; ERP = {:.1f}% &nbsp;|&nbsp; g = Geçmiş temettü büyümesi (yoksa {:.1f}%)
        </span>
    </div>
    """.format(rf * 100, erp * 100, g_default * 100), unsafe_allow_html=True)

    # DDM cards
    cols = st.columns(min(len(ddm_df), 3))
    for i, (ticker, row) in enumerate(ddm_df.iterrows()):
        fair = row["DDM Adil Değer"]
        price = row["Mevcut Fiyat"]
        upside = row["Potansiyel (%)"]

        if fair is None:
            with cols[i % 3]:
                st.markdown(f"""
                <div class="ddm-result overvalued">
                    <div class="ticker-label">{ticker}</div>
                    <div style="color:#ff5252; font-family:'IBM Plex Mono',monospace; font-size:1rem; margin-top:6px;">
                        ⚠ Temettü verisi yetersiz — DDM uygulanamaz
                    </div>
                </div>
                """, unsafe_allow_html=True)
            continue

        is_over = upside is not None and upside < 0
        card_class = "ddm-result overvalued" if is_over else "ddm-result"
        arrow = "🔴 Aşırı Değerli" if is_over else "🟢 İskontolu"
        up_color = "#ff5252" if is_over else "#00e676"

        with cols[i % 3]:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="ticker-label">{ticker}</div>
                <div class="fair-value">{fair:,.2f} TL</div>
                <div style="font-size:0.78rem; color:#4a7a9b; margin-top:4px;">DDM Adil Değer</div>
                <div style="margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:6px; font-family:'IBM Plex Mono',monospace; font-size:0.80rem;">
                    <div><span style="color:#4a7a9b;">Mevcut:</span> <span style="color:#e8f4ff;">{price:,.2f} TL</span></div>
                    <div><span style="color:#4a7a9b;">D₀:</span> <span style="color:#e8f4ff;">{row['D₀'] or '—'} TL</span></div>
                    <div><span style="color:#4a7a9b;">g:</span> <span style="color:#e8f4ff;">%{row['g (%)']}</span></div>
                    <div><span style="color:#4a7a9b;">Ke:</span> <span style="color:#e8f4ff;">%{row['Ke (%)']}</span></div>
                </div>
                <div style="margin-top:10px; font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:1.0rem; color:{up_color};">
                    {arrow} &nbsp; {f'%{upside:+.1f}' if upside else '—'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # DDM summary table
    st.markdown('<div class="section-header">DDM Özet Tablosu</div>', unsafe_allow_html=True)
    display_ddm = ddm_df.copy()
    display_ddm["Mevcut Fiyat"] = display_ddm["Mevcut Fiyat"].map(
        lambda x: f"{x:,.2f} TL" if pd.notna(x) else "—")
    display_ddm["DDM Adil Değer"] = display_ddm["DDM Adil Değer"].map(
        lambda x: f"{x:,.2f} TL" if pd.notna(x) else "DDM Uygulanamaz")

    def color_upside(val):
        if pd.isna(val) or val == "—": return ""
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            if v >= 20: return "background-color:#0a2a12; color:#00e676"
            if v >= 0:  return "background-color:#0a1e0a; color:#80e676"
            if v >= -20: return "background-color:#2a1206; color:#f39c12"
            return "background-color:#2a0a0a; color:#ff5252"
        except: return ""

    display_ddm["Potansiyel (%)"] = display_ddm["Potansiyel (%)"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")

    st.dataframe(
        display_ddm.style.applymap(color_upside, subset=["Potansiyel (%)"]) \
            .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "0.85rem"}),
        use_container_width=True
    )

    st.markdown("""
    <div style="background:#0a0f1a; border:1px solid #1a3a5c; border-radius:4px; padding:10px 14px; margin-top:12px;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#4a7a9b;">
        ⚠ DDM Sınırlaması: g ≥ Ke olduğunda model çalışmaz.
        Yüksek büyüme döneminde DDM tek başına yeterli değil; FD/FAVÖK ve DCF ile desteklenmeli.
        Büyüme oranı TL cinsinden nominal GSYH büyümesinin altında olmalıdır.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — FINANCIAL RATIOS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Karlılık Rasyoları</div>', unsafe_allow_html=True)

    margin_cols = ["Brüt Marj %", "FAVÖK Marjı %", "Net Kar Marjı %"]
    margin_data = valid[margin_cols].dropna(how="all")

    if not margin_data.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("#070d18")
        ax.set_facecolor("#0a1525")

        x = np.arange(len(margin_data))
        w = 0.25
        colors_margin = ["#5ab4e0", "#00e676", "#f39c12"]

        for k, (col, color) in enumerate(zip(margin_cols, colors_margin)):
            vals = margin_data[col].fillna(0)
            bars = ax.bar(x + k * w, vals, w, label=col, color=color, alpha=0.85, edgecolor="none")
            for bar, val in zip(bars, vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}%", ha="center", va="bottom",
                            color="#c8d8e8", fontsize=8, fontfamily="monospace")

        ax.set_xticks(x + w)
        ax.set_xticklabels(margin_data.index, color="#8ab0c8", fontsize=10)
        ax.tick_params(colors="#4a7a9b")
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", alpha=0.1, color="#2a4a6a")
        ax.set_ylabel("Marj (%)", color="#4a7a9b", fontsize=9)
        ax.legend(framealpha=0.1, facecolor="#0a1525", edgecolor="#1a3a5c",
                  labelcolor="#c8d8e8", fontsize=9)
        ax.set_title("Karlılık Marjları Karşılaştırması", color="#4a7a9b", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Valuation multiples chart ──
    st.markdown('<div class="section-header">Değerleme Çarpanları</div>', unsafe_allow_html=True)
    mult_data = valid[["F/K", "FD/FAVÖK", "PD/DD"]].dropna(how="all")

    if not mult_data.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#070d18")

        for ax, col, color, title in zip(
            axes,
            ["F/K", "FD/FAVÖK", "PD/DD"],
            ["#5ab4e0", "#00e676", "#f39c12"],
            ["F/K (P/E)", "FD/FAVÖK", "PD/DD (P/B)"]
        ):
            ax.set_facecolor("#0a1525")
            data_col = mult_data[col].dropna().sort_values()
            ax.barh(data_col.index, data_col.values, color=color, alpha=0.8, edgecolor="none", height=0.5)
            for i, (idx, val) in enumerate(data_col.items()):
                ax.text(val + 0.05, i, f"{val:.1f}x", va="center",
                        color="#e8f4ff", fontsize=9, fontfamily="monospace")
            ax.spines[:].set_visible(False)
            ax.tick_params(colors="#8ab0c8", labelsize=9)
            ax.grid(axis="x", alpha=0.1, color="#2a4a6a")
            ax.set_title(title, color="#4a7a9b", fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Full ratios table ──
    st.markdown('<div class="section-header">Tüm Finansal Rasyolar</div>', unsafe_allow_html=True)
    ratio_cols = ["Şirket", "F/K", "FD/FAVÖK", "PD/DD", "ROE %", "ROA %",
                  "Net Kar Marjı %", "FAVÖK Marjı %", "Borç/Özkaynak", "Cari Oran",
                  "Temettü V. %", "Beta"]
    ratio_table = valid[ratio_cols].copy()

    def highlight_ratio(col):
        styles = []
        for val in col:
            if pd.isna(val):
                styles.append("")
            elif col.name in ["ROE %", "ROA %", "Net Kar Marjı %", "FAVÖK Marjı %", "Temettü V. %"]:
                styles.append("color: #00e676" if val > 0 else "color: #ff5252")
            elif col.name in ["F/K", "FD/FAVÖK", "PD/DD", "Borç/Özkaynak"]:
                median = col.median()
                styles.append("color: #00e676" if val < median else "color: #f39c12")
            else:
                styles.append("")
        return styles

    fmt = {c: "{:.2f}" for c in ratio_cols if c != "Şirket"}
    fmt["ROE %"] = "{:.1f}%"
    fmt["ROA %"] = "{:.1f}%"
    fmt["Net Kar Marjı %"] = "{:.1f}%"
    fmt["FAVÖK Marjı %"] = "{:.1f}%"
    fmt["Temettü V. %"] = "{:.2f}%"

    st.dataframe(
        ratio_table.style.apply(highlight_ratio).format(fmt, na_rep="—")
            .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "0.82rem"}),
        use_container_width=True
    )


# ══════════════════════════════════════════════
# TAB 4 — DuPont Analysis
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">DuPont Analizi — ROE Ayrıştırması</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1f35; border:1px solid #1a3a5c; border-radius:6px; padding:12px 18px; margin-bottom:16px; font-family:'IBM Plex Mono',monospace;">
        <span style="color:#4a7a9b; font-size:0.72rem;">3 FAKTÖRLÜ DUPONT</span><br>
        <span style="color:#e8f4ff; font-size:1.0rem;">ROE = Net Kâr Marjı × Aktif Devir Hızı × Finansal Kaldıraç</span>
    </div>
    """, unsafe_allow_html=True)

    dupont_rows = []
    for ticker, row in valid.iterrows():
        roe = row["ROE %"] / 100 if pd.notna(row["ROE %"]) else None
        net_margin = row["Net Kar Marjı %"] / 100 if pd.notna(row["Net Kar Marjı %"]) else None
        # Approx: D/E ratio to get leverage (1 + D/E)
        de = row["Borç/Özkaynak"] / 100 if pd.notna(row["Borç/Özkaynak"]) else None
        leverage = (1 + de) if de is not None else None
        # Asset turnover = ROA / Net Margin
        roa = row["ROA %"] / 100 if pd.notna(row["ROA %"]) else None
        asset_turnover = (roa / net_margin) if (roa and net_margin and net_margin != 0) else None
        # Reconstructed ROE
        rec_roe = (net_margin * asset_turnover * leverage) if all(
            x is not None for x in [net_margin, asset_turnover, leverage]) else None

        dupont_rows.append({
            "Ticker": ticker,
            "Net Kâr Marjı": f"{net_margin * 100:.1f}%" if net_margin else "—",
            "Aktif Devir Hızı": f"{asset_turnover:.2f}x" if asset_turnover else "—",
            "Finansal Kaldıraç": f"{leverage:.2f}x" if leverage else "—",
            "Hesaplanan ROE": f"{rec_roe * 100:.1f}%" if rec_roe else "—",
            "Gerçek ROE": f"{row['ROE %']:.1f}%" if pd.notna(row["ROE %"]) else "—",
        })

    dupont_display = pd.DataFrame(dupont_rows).set_index("Ticker")
    st.dataframe(
        dupont_display.style.set_properties(
            **{"font-family": "IBM Plex Mono, monospace", "font-size": "0.88rem"}),
        use_container_width=True
    )

    # ROE components stacked bar
    st.markdown('<div class="section-header">ROE Bileşen Görselleştirmesi</div>', unsafe_allow_html=True)

    roe_data = valid[["ROE %", "ROA %", "Net Kar Marjı %"]].dropna(how="all")
    if not roe_data.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#070d18")
        ax.set_facecolor("#0a1525")

        x = np.arange(len(roe_data))
        w = 0.25
        for k, (col, color) in enumerate(zip(
            ["ROE %", "ROA %", "Net Kar Marjı %"],
            ["#00e676", "#5ab4e0", "#f39c12"]
        )):
            vals = roe_data[col].fillna(0)
            ax.bar(x + k * w, vals, w, label=col, color=color, alpha=0.8, edgecolor="none")

        ax.set_xticks(x + w)
        ax.set_xticklabels(roe_data.index, color="#8ab0c8", fontsize=10)
        ax.tick_params(colors="#4a7a9b")
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", alpha=0.1, color="#2a4a6a")
        ax.set_ylabel("%", color="#4a7a9b", fontsize=9)
        ax.legend(framealpha=0.1, facecolor="#0a1525", edgecolor="#1a3a5c",
                  labelcolor="#c8d8e8", fontsize=9)
        ax.set_title("ROE / ROA / Net Kâr Marjı Karşılaştırması", color="#4a7a9b", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Kaldıraç riski
    st.markdown('<div class="section-header">Kaldıraç & Risk Profili</div>', unsafe_allow_html=True)
    lev_cols = ["Borç/Özkaynak", "Cari Oran", "Beta"]
    lev_data = valid[lev_cols + ["Şirket"]].copy()

    def style_leverage(val, col):
        if pd.isna(val): return ""
        if col == "Borç/Özkaynak":
            return "color:#00e676" if val < 50 else "color:#f39c12" if val < 150 else "color:#ff5252"
        if col == "Cari Oran":
            return "color:#00e676" if val > 1.5 else "color:#f39c12" if val > 1.0 else "color:#ff5252"
        if col == "Beta":
            return "color:#00e676" if val < 0.8 else "color:#f39c12" if val < 1.2 else "color:#ff5252"
        return ""

    styled_lev = lev_data.style
    for col in lev_cols:
        styled_lev = styled_lev.applymap(lambda v: style_leverage(v, col), subset=[col])
    styled_lev = styled_lev.format(
        {"Borç/Özkaynak": "{:.1f}", "Cari Oran": "{:.2f}x", "Beta": "{:.2f}"},
        na_rep="—"
    ).set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "0.85rem"})
    st.dataframe(styled_lev, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — PRICE & DIVIDEND HISTORY
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Fiyat Gelişimi (2 Yıl)</div>', unsafe_allow_html=True)

    selected = st.selectbox("Hisse seçin:", valid.index.tolist())

    if selected in hist_data and not hist_data[selected].empty:
        hist = hist_data[selected].copy()

        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor("#070d18")
        gs = GridSpec(3, 1, figure=fig, hspace=0.05, height_ratios=[3, 1, 1])

        # Price chart
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("#0a1525")
        close = hist["Close"]
        ax1.plot(close.index, close, color="#5ab4e0", linewidth=1.5, label="Kapanış")
        ax1.fill_between(close.index, close, close.min(), alpha=0.08, color="#5ab4e0")

        # 50 & 200 MA
        if len(close) >= 50:
            ax1.plot(close.index, close.rolling(50).mean(), color="#f39c12",
                     linewidth=1, linestyle="--", alpha=0.7, label="MA50")
        if len(close) >= 200:
            ax1.plot(close.index, close.rolling(200).mean(), color="#00e676",
                     linewidth=1, linestyle="--", alpha=0.7, label="MA200")

        ax1.spines[:].set_visible(False)
        ax1.tick_params(colors="#4a7a9b", labelsize=8)
        ax1.set_xticklabels([])
        ax1.grid(alpha=0.08, color="#2a4a6a")
        ax1.set_ylabel("Fiyat (TL)", color="#4a7a9b", fontsize=9)
        ax1.legend(framealpha=0.1, facecolor="#0a1525", edgecolor="#1a3a5c",
                   labelcolor="#c8d8e8", fontsize=8)
        ax1.set_title(f"{selected} — Fiyat & Teknik Göstergeler", color="#4a7a9b", fontsize=10)

        # Volume
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor("#0a1525")
        vol_colors = ["#00e676" if c >= o else "#ff5252"
                      for c, o in zip(hist["Close"], hist["Open"])]
        ax2.bar(hist.index, hist["Volume"], color=vol_colors, alpha=0.6, width=1)
        ax2.spines[:].set_visible(False)
        ax2.tick_params(colors="#4a7a9b", labelsize=7)
        ax2.set_xticklabels([])
        ax2.set_ylabel("Hacim", color="#4a7a9b", fontsize=8)

        # RSI
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor("#0a1525")
        if len(close) >= 20:
            d = close.diff()
            gain = d.clip(lower=0).rolling(14).mean()
            loss = (-d.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            ax3.plot(rsi_series.index, rsi_series, color="#a78bfa", linewidth=1.2)
            ax3.axhline(70, color="#ff5252", linestyle="--", alpha=0.4, linewidth=0.8)
            ax3.axhline(30, color="#00e676", linestyle="--", alpha=0.4, linewidth=0.8)
            ax3.fill_between(rsi_series.index, rsi_series, 70,
                             where=rsi_series >= 70, alpha=0.15, color="#ff5252")
            ax3.fill_between(rsi_series.index, rsi_series, 30,
                             where=rsi_series <= 30, alpha=0.15, color="#00e676")
            ax3.set_ylim(0, 100)
        ax3.spines[:].set_visible(False)
        ax3.tick_params(colors="#4a7a9b", labelsize=7)
        ax3.set_ylabel("RSI(14)", color="#4a7a9b", fontsize=8)
        ax3.grid(alpha=0.06, color="#2a4a6a")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(f"{selected} için fiyat verisi bulunamadı.")

    # ── Dividend history chart ──
    st.markdown('<div class="section-header">Temettü Geçmişi</div>', unsafe_allow_html=True)

    if selected in div_data and not div_data[selected].empty:
        divs = div_data[selected].copy()
        divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
        annual = divs.groupby(divs.index.year).sum()

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#070d18")
        ax.set_facecolor("#0a1525")

        bars = ax.bar(annual.index.astype(str), annual.values, color="#00e676",
                      alpha=0.8, edgecolor="none", width=0.55)
        for bar, val in zip(bars, annual.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + annual.max() * 0.01,
                    f"{val:.2f} TL", ha="center", va="bottom",
                    color="#e8f4ff", fontsize=10, fontweight="bold", fontfamily="monospace")

        ax.spines[:].set_visible(False)
        ax.tick_params(colors="#8ab0c8", labelsize=10)
        ax.grid(axis="y", alpha=0.08, color="#2a4a6a")
        ax.set_ylabel("Hisse Başı Temettü (TL)", color="#4a7a9b", fontsize=9)
        ax.set_title(f"{selected} — Yıllık Temettü Geçmişi", color="#4a7a9b", fontsize=10)

        # Growth annotation
        if len(annual) >= 2:
            cagr = (annual.iloc[-1] / annual.iloc[0]) ** (1 / (len(annual) - 1)) - 1
            ax.annotate(f"CAGR: %{cagr * 100:.1f}",
                        xy=(0.98, 0.92), xycoords="axes fraction",
                        ha="right", color="#f39c12", fontsize=10,
                        fontfamily="monospace", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(f"{selected} için temettü verisi bulunamadı.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#2a4a6a; padding:10px 0;">
    Veri: Yahoo Finance • DDM: Gordon Growth Model • Yatırım tavsiyesi değildir • 2025
</div>
""", unsafe_allow_html=True)
