import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Dividend & Value Screener", layout="wide", page_icon="trophy")
st.title("Smart Stock Screener – Peer-Relative Scoring")
st.markdown("**P/E • P/B • ROE • Profit Margin → True 0–100 Score**")

# ================================
# PREDEFINED HIGH-QUALITY SETS
# ================================
predefined_sets = {
    "Top 5 Royalty/Streaming": ["WPM", "FNV", "RGLD", "OR", "SAND"],
    "Top 5 Triple-Net REITs":   ["O", "NNN", "WPC", "SRC", "ADC"],
    "Top 5 Tech Giants":        ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Top 5 Gold Miners":        ["AEM", "NEM", "GOLD", "KGC", "PAAS"],
    "Reliable Dividend Mix":    ["AEM", "WPM", "FNV", "O", "NNN", "WPC"]
}

# Initialize session state for persistent ticker list
if "tickers" not in st.session_state:
    st.session_state.tickers = predefined_sets["Reliable Dividend Mix"]

# ================================
# USER INPUT – ADD TO LIST
# ================================
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    add_manual = st.text_input(
        "Add tickers (space/comma separated):",
        placeholder="e.g. VICI NVDA TSLA"
    )
    if add_manual.strip():
        new_ticks = [t.strip().upper() for t in add_manual.replace(",", " ").split() if t.strip()]
        current = set(st.session_state.tickers)
        st.session_state.tickers = sorted(list(current.union(new_ticks)))

with col2:
    selected_set = st.selectbox(
        "Or load a predefined set → adds to current list",
        options=[""] + list(predefined_sets.keys())
    )
    if selected_set:
        st.session_state.tickers = sorted(
            list(set(st.session_state.tickers).union(predefined_sets[selected_set]))
        )
        st.success(f"Added {selected_set}")

with col3:
    st.write("**Current Watchlist:**")
    if st.session_state.tickers:
        st.write(", ".join(st.session_state.tickers))
        if st.button("Clear All"):
            st.session_state.tickers = []
            st.rerun()
    else:
        st.info("List empty – add tickers above")

if not st.session_state.tickers:
    st.stop()

tickers = st.session_state.tickers

# ================================
# FETCH DATA
# ================================
@st.cache_data(ttl=1800)
def fetch_data(tickers):
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            # RSI
            hist = stock.history(period="60d")
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = round(rsi.iloc[-1], 1) if len(rsi) > 0 else None

            # Dividend
            divs = stock.dividends
            recent = divs[divs.index > divs.index.max() - pd.Timedelta(days=400)]
            last_div = recent.iloc[-1] if not recent.empty else 0
            annual_div = round(last_div * 4, 3) if last_div > 0 else 0
            yield_pct = round((annual_div / price) * 100, 2) if price and annual_div > 0 else 0

            data.append({
                "Ticker": t,
                "Name": info.get("longName", t).split(" Corporation")[0].split(" Inc")[0],
                "Price": price,
                "RSI (14)": current_rsi,
                "Yield %": yield_pct,
                "Annual Div $": annual_div,
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except Exception as e:
            st.warning(f"{t}: {e}")
            data.append({"Ticker": t, "Name": "Error", "Price": None, "RSI (14)": None, "Yield %": 0,
                         "Annual Div $": 0, "P/E": None, "P/B": None, "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# ================================
# SMART PEER-RELATIVE SCORING (fixed & perfect)
# ================================
def calculate_fundamental_score(row, v_pe, v_pb, v_roe, v_margin):
    comp = []
    if row["P/E"] and row["P/E"] > 0 and len(v_pe) > 1:
        pe_score = 100 * (max(v_pe) - row["P/E"]) / (max(v_pe) - min(v_pe))
        comp.append((pe_score, 0.30))
    if row["P/B"] and row["P/B"] > 0 and len(v_pb) > 1:
        pb_score = 100 * (max(v_pb) - row["P/B"]) / (max(v_pb) - min(v_pb))
        comp.append((pb_score, 0.25))
    if pd.notna(row["ROE %"]) and len(v_roe) > 1:
        roe_score = 100 * (row["ROE %"] - min(v_roe)) / (max(v_roe) - min(v_roe))
        comp.append((roe_score, 0.25))
    if pd.notna(row["Profit Margin %"]) and len(v_margin) > 1:
        m_score = 100 * (row["Profit Margin %"] - min(v_margin)) / (max(v_margin) - min(v_margin))
        comp.append((m_score, 0.20))
    if not comp: return None
    total_w = sum(w for _, w in comp)
    return round(sum(s * w for s, w in comp) / total_w, 1)

valid_pe = [v for v in df["P/E"] if v and v > 0]
valid_pb = [v for v in df["P/B"] if v and v > 0]
valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

df["Fundamental Score"] = df.apply(
    lambda row: calculate_fundamental_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1
)

# ================================
# 1. VALUATION TABLE
# ================================
st.subheader("1. Relative Fair Value (P/E + P/B)")
peer_pe = df["P/E"].median()
peer_pb = df["P/B"].median()
val_rows = []
for idx, row in df.iterrows():
    rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else None
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else None
    fair = np.nanmean([rel_pe, rel_pb]) if rel_pe or rel_pb else None
    upside = (fair / row["Price"] - 1) * 100 if fair and row["Price"] else None
    val_rows.append({"Ticker": idx, "Price": row["Price"], "P/E Fair": rel_pe, "P/B Fair": rel_pb,
                     "Fair Value": fair, "Upside %": upside})

val_df = pd.DataFrame(val_rows).round(2).set_index("Ticker")
st.dataframe(val_df.style.format({"Price": "${:,.2f}", "P/E Fair": "${:,.2f}", "P/B Fair": "${:,.2f}",
                                  "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"}, na_rep="—")
             .background_gradient(subset=["Upside %"], cmap="RdYlGn"), use_container_width=True)

# ================================
# 2. CLEAN SCORE TABLE
# ================================
st.subheader("2. Technical + Income + Smart Score")
display = df[["RSI (14)", "Yield %", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]].round(2)
st.dataframe(display.style.format({"RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "P/E": "{:.1f}",
                                   "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"}, na_rep="—"),
             use_container_width=True)

# ================================
# 3. RANKING CHART – GREEN BARS FIXED
# ================================
st.subheader("3. Best Opportunities – Ranked by Smart Score")
scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)

if not scored.empty:
    scores = scored["Fundamental Score"]
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]  # Real green!

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(scored))))
    ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100) – Higher = Better")
    ax.set_title("Green = Top Tier Value • Yellow = Fair • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, s in enumerate(scores[::-1]):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold', color="black")
    st.pyplot(fig)
else:
    st.warning("Not enough data for scoring")

st.success("Manual add + Predefined sets + Persistent list + Green bars fixed")
st.caption("Data: Yahoo Finance • Real-time • Not financial advice • 2025")
