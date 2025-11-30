import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Dividend & Value Screener", layout="wide", page_icon="trophy")
st.title("Smart Stock Screener – Accurate Yield + Peer Scoring")
st.markdown("**Official dividend yield • P/E • P/B • ROE • Margin → True 0–100 Score**")

# ================================
# PREDEFINED SETS
# ================================
predefined_sets = {
    "Top 5 Royalty/Streaming": ["WPM", "FNV", "RGLD", "OR", "SAND"],
    "Top 5 Triple-Net REITs":   ["O", "NNN", "WPC", "ADC", "SRC"],
    "Top 5 Tech Giants":        ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Top 5 Gold Miners":        ["AEM", "NEM", "GOLD", "KGC", "PAAS"],
    "Reliable Dividend Mix":    ["AEM", "WPM", "FNV", "O", "NNN", "WPC"]
}

if "tickers" not in st.session_state:
    st.session_state.tickers = predefined_sets["Reliable Dividend Mix"]

# ================================
# USER INPUT
# ================================
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    add_manual = st.text_input("Add tickers (space/comma):", placeholder="VICI NVDA TSLA")
    if add_manual.strip():
        new = [t.strip().upper() for t in add_manual.replace(",", " ").split() if t.strip()]
        st.session_state.tickers = sorted(list(set(st.session_state.tickers) | set(new)))

with col2:
    selected_set = st.selectbox("Load predefined set → adds to list", [""] + list(predefined_sets.keys()))
    if selected_set:
        st.session_state.tickers = sorted(list(set(st.session_state.tickers) | set(predefined_sets[selected_set])))
        st.success(f"Added {selected_set}")

with col3:
    st.write("**Current Watchlist:**")
    if st.session_state.tickers:
        st.write(", ".join(st.session_state.tickers))
        if st.button("Clear All"):
            st.session_state.tickers = []
            st.rerun()
    else:
        st.info("Empty – add stocks above")

if not st.session_state.tickers:
    st.stop()

tickers = st.session_state.tickers

# ================================
# FETCH DATA – OFFICIAL YIELD
# ================================
@st.cache_data(ttl=1800)
def fetch_data(tickers):
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            # Official trailing 12-month dividend yield
            yield_official = info.get("trailingAnnualDividendYield")
            yield_pct = round(yield_official * 100, 2) if yield_official else 0.0

            # Trailing 12-month total dividend
            annual_div = info.get("trailingAnnualDividendRate") or 0.0

            # RSI
            hist = stock.history(period="60d")
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).rolling(10).mean()
            loss = -delta.clip(upper=0).rolling(10).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = round(rsi.iloc[-1], 1) if len(rsi) > 0 else None

            data.append({
                "Ticker": t,
                "Name": info.get("longName", t).split(" Corporation")[0].split(" Inc")[0],
                "Price": price,
                "RSI (10)": current_rsi,
                "Yield %": yield_pct,
                "Annual Div $": round(annual_div, 3),
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except Exception as e:
            st.warning(f"{t}: {e}")
            data.append({"Ticker": t, "Name": "Error", "Price": None, "RSI (14)": None,
                         "Yield %": 0.0, "Annual Div $": 0.0, "P/E": None, "P/B": None,
                         "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# ================================
# SMART SCORING (perfect)
# ================================
def calc_score(row, v_pe, v_pb, v_roe, v_margin):
    comp = []
    if row["P/E"] and row["P/E"] > 0 and len(v_pe) > 1:
        comp.append((100 * (max(v_pe) - row["P/E"]) / (max(v_pe) - min(v_pe)), 0.30))
    if row["P/B"] and row["P/B"] > 0 and len(v_pb) > 1:
        comp.append((100 * (max(v_pb) - row["P/B"]) / (max(v_pb) - min(v_pb)), 0.25))
    if pd.notna(row["ROE %"]) and len(v_roe) > 1:
        comp.append((100 * (row["ROE %"] - min(v_roe)) / (max(v_roe) - min(v_roe)), 0.25))
    if pd.notna(row["Profit Margin %"]) and len(v_margin) > 1:
        comp.append((100 * (row["Profit Margin %"] - min(v_margin)) / (max(v_margin) - min(v_margin)), 0.20))
    if not comp: return None
    return round(sum(s * w for s, w in comp) / sum(w for _, w in comp), 1)

valid_pe = [v for v in df["P/E"] if v and v > 0]
valid_pb = [v for v in df["P/B"] if v and v > 0]
valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

df["Fundamental Score"] = df.apply(lambda row: calc_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1)

# ================================
# 1. VALUATION TABLE
# ================================
st.subheader("1. Relative Fair Value")
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
# 2. CLEAN TABLE WITH OFFICIAL YIELD
# ================================
st.subheader("2. Technical + Income + Smart Score")
display = df[["RSI (10)", "Yield %", "Annual Div $", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]].round(2)
st.dataframe(display.style.format({"RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "Annual Div $": "${:.3f}",
                                   "P/E": "{:.1f}", "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"},
                                  na_rep="—"), use_container_width=True)

# ================================
# 3. RANKING CHART
# ================================
st.subheader("3. Best Opportunities – Ranked by Smart Score")
scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)
if not scored.empty:
    scores = scored["Fundamental Score"]
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(scored))))
    ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100)")
    ax.set_title("Green = Top Tier • Yellow = Fair • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, s in enumerate(scores[::-1]):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold')
    st.pyplot(fig)
else:
    st.warning("Not enough data")

st.success("Official dividend yield • No more ×4 confusion • Clean & accurate")
st.caption("Data: Yahoo Finance (official trailing yield) • Real-time • Not advice • 2025")
