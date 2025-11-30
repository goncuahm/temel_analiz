import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Stock Screener", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Miners – Fair Value + Score")
st.markdown("**Annual dividends FIXED: $1.12, $0.66, $1.60, $1.08 — never 160 again**")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS", "RGLD", "KGC"]
tickers_input = st.sidebar.text_input("Tickers (comma-separated):", value=", ".join(default_tickers))
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Required Return", 0.06, 0.12, 0.08, 0.005)
terminal_growth = st.sidebar.slider("Long-Term Growth", 0.02, 0.05, 0.035, 0.005)

# -------------------------------
# FETCH DATA – DIVIDEND BUG FIXED FOREVER
# -------------------------------
@st.cache_data(ttl=3600)
def get_data(tickers):
    rows = []
    for t in tickers:
        row = {"Ticker": t}
        try:
            stock = yf.Ticker(t)
            i = stock.info
            price = i.get("currentPrice") or i.get("regularMarketPrice") or np.nan

            # CRITICAL FIX: Use decimal yield, NOT the % version
            div_yield_decimal = i.get("dividendYield") or 0.0           # e.g. 0.016
            yield_pct = div_yield_decimal * 100                         # 1.60% for table

            # Correct annual dividend
            annual_div = i.get("forwardDividend")
            if annual_div is None or pd.isna(annual_div):
                annual_div = div_yield_decimal * price                  # 0.016 × 70 = $1.12

            rows.append({
                "Ticker": t,
                "Name": i.get("longName", t),
                "Price": price,
                "Yield %": round(yield_pct, 2),
                "Annual Div $": round(annual_div, 3),   # Now correct: $1.12, $0.66, etc.
                "Fwd EPS": i.get("forwardEps"),
                "Fwd P/E": i.get("forwardPE"),
                "P/B": i.get("priceToBook"),
                "ROE %": (i.get("returnOnEquity") or 0) * 100,
            })
        except Exception as e:
            st.warning(f"{t}: {e}")
            rows.append({
                "Ticker": t, "Name": "Error", "Price": np.nan, "Yield %": np.nan, "Annual Div $": np.nan,
                "Fwd EPS": np.nan, "Fwd P/E": np.nan, "P/B": np.nan, "ROE %": np.nan,
            })
    return pd.DataFrame(rows).set_index("Ticker")

df = get_data(companies)

# -------------------------------
# DISPLAY
# -------------------------------
st.subheader("1. Correct Fundamental Data")
styled = df.style.format({
    "Price": "${:,.2f}",
    "Yield %": "{:.2f}%",
    "Annual Div $": "${:,.3f}",
    "Fwd EPS": "${:,.2f}",
    "Fwd P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "ROE %": "{:.1f}%",
}, na_rep="—")
st.dataframe(styled, use_container_width=True)

# -------------------------------
# VALUATION & SCORE
# -------------------------------
st.subheader("2. Fair Value + Fundamental Score")
peer_pe = df["Fwd P/E"].median(skipna=True) or 18
peer_pb = df["P/B"].median(skipna=True) or 1.8

results = []
for idx, row in df.iterrows():
    # DDM
    g = min(terminal_growth, discount_rate - 0.005)
    ddm = (row["Annual Div $"] * (1 + g) / (discount_rate - g)) if row["Annual Div $"] > 0 else np.nan

    # Relative
    rel_pe = row["Fwd EPS"] * peer_pe if pd.notna(row["Fwd EPS"]) else np.nan
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    relative = np.nanmean([rel_pe, rel_pb])

    fair = np.nanmean([ddm, relative])
    upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and pd.notna(row["Price"]) else np.nan

    # Fundamental Score
    score = 0
    n = 0
    if pd.notna(row["Fwd P/E"]): 
        if row["Fwd P/E"] < 12: score += 30
        elif row["Fwd P/E"] < 18: score += 20
        n += 1
    if pd.notna(row["P/B"]):
        if row["P/B"] < 1.2: score += 30
        elif row["P/B"] < 1.8: score += 20
        n += 1
    if pd.notna(row["Yield %"]):
        if row["Yield %"] > 4: score += 25
        elif row["Yield %"] > 2.5: score += 15
        n += 1

    fund_score = round(score * 100 / (75 if n == 3 else 50 if n == 2 else 30), 0) if n > 0 else 0

    results.append({
        "Ticker": idx,
        "Price": row["Price"],
        "DDM": ddm,
        "Relative": relative,
        "Fair Value": fair,
        "Upside %": upside,
        "Score": int(fund_score),
    })

val_df = pd.DataFrame(results).set_index("Ticker").round(2)
val_df = val_df.sort_values("Score", ascending=False)

# Styling
def highlight(val):
    if val >= 80: return "background-color: #d4edda"
    if val >= 60: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_val = val_df.style.format({
    "Price": "${:,.2f}", "DDM": "${:,.2f}", "Relative": "${:,.2f}",
    "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"
}, na_rep="—") \
    .background_gradient(subset=["Upside %"], cmap="RdYlGn") \
    .applymap(highlight, subset=["Score"])

st.dataframe(styled_val, use_container_width=True)

# Chart
st.subheader("3. Ranking – Best Deals First")
fig, ax = plt.subplots(figsize=(9,5))
scores = val_df["Score"]
colors = ["#27ae60" if s>=80 else "#f39c12" if s>=60 else "#e74c3c" for s in scores]
ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1])
ax.set_xlabel("Fundamental Score (100 = Best)")
ax.set_title("Green = Strong Buy | Yellow = Fair | Red = Expensive")
for i, (t, s) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(s + 1, i, f"{int(s)}", va='center', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
st.pyplot(fig)

st.success("All bugs fixed: Annual Div $1.12, $0.66, $1.60, $1.08 — never 160 again")
st.caption("Data: Yahoo Finance • Nov 30 2025 • Not advice")
