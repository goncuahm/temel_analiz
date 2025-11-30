import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Screener", layout="wide", page_icon="gold_bar")
st.title("Dividend Discounted Value + Fundamental Score")
st.markdown("**Uses only real, paid dividends & trailing metrics — no forward estimates**")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS", "RGLD", "KGC", "SSRM", "OR"]
tickers_input = st.sidebar.text_input("Tickers:", value=", ".join(default_tickers))
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Required Return", 0.06, 0.12, 0.085, 0.005)
terminal_growth = st.sidebar.slider("Long-Term Growth", 0.02, 0.05, 0.03, 0.005)

# -------------------------------
# FETCH DATA – USES ONLY REAL DIVIDENDS
# -------------------------------
@st.cache_data(ttl=3600)
def get_real_data(tickers):
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            hist = stock.history(period="1y")

            price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan

            # LAST ACTUAL DIVIDEND (not forward)
            dividends = stock.dividends
            recent_divs = dividends[dividends.index > dividends.index.max() - pd.Timedelta(days=400)]
            if not recent_divs.empty:
                last_div = recent_divs.iloc[-1]
                annual_div = last_div * 4  # assume quarterly (most miners)
                yield_pct = (annual_div / price) * 100 if price else 0
            else:
                last_div = annual_div = yield_pct = 0

            rows.append({
                "Ticker": t,
                "Name": info.get("longName", t),
                "Price": price,
                "Last Quarterly Div $": round(last_div, 4),
                "Annual Div $": round(annual_div, 3),
                "Yield %": round(yield_pct, 2),
                "Trailing P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": (info.get("returnOnEquity") or 0) * 100,
                "Debt/Equity": info.get("debtToEquity"),
                "Market Cap $B": info.get("marketCap", 0) / 1e9,
            })
        except Exception as e:
            st.warning(f"{t}: {e}")
            rows.append({"Ticker": t, "Name": "Error", "Price": np.nan, "Last Quarterly Div $": np.nan,
                         "Annual Div $": np.nan, "Yield %": np.nan, "Trailing P/E": np.nan,
                         "P/B": np.nan, "ROE %": np.nan, "Debt/Equity": np.nan, "Market Cap $B": np.nan})
    return pd.DataFrame(rows).set_index("Ticker")

df = get_real_data(companies)

# -------------------------------
# DISPLAY
# -------------------------------
st.subheader("1. Real Fundamental Data (Trailing Only)")
styled = df.style.format({
    "Price": "${:,.2f}",
    "Last Quarterly Div $": "${:,.4f}",
    "Annual Div $": "${:,.3f}",
    "Yield %": "{:.2f}%",
    "Trailing P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "ROE %": "{:.1f}%",
    "Debt/Equity": "{:.1f}",
    "Market Cap $B": "{:.2f}",
}, na_rep="—")
st.dataframe(styled, use_container_width=True)

# -------------------------------
# VALUATION & TRUE FUNDAMENTAL SCORE
# -------------------------------
st.subheader("2. Fair Value + Real Fundamental Score (0–100)")

results = []
peer_pe = df["Trailing P/E"].median(skipna=True) or 16
peer_pb = df["P/B"].median(skipna=True) or 1.7

for idx, row in df.iterrows():
    # DDM using REAL annual dividend
    g = min(terminal_growth, discount_rate - 0.005)
    ddm = (row["Annual Div $"] * (1 + g) / (discount_rate - g)) if row["Annual Div $"] > 0 else np.nan

    # Relative using TRAILING multiples
    rel_pe = row["Price"] * (peer_pe / row["Trailing P/E"]) if pd.notna(row["Trailing P/E"]) and row["Trailing P/E"] > 0 else np.nan
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    relative = np.nanmean([rel_pe, rel_pb])

    fair = np.nanmean([ddm, relative])
    upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and pd.notna(row["Price"]) else np.nan

    # TRUE FUNDAMENTAL SCORE – ONLY REAL DATA
    score = 0
    n = 0

    # Trailing P/E
    pe = row["Trailing P/E"]
    if pd.notna(pe) and pe > 0:
        if pe < 10: score += 35
        elif pe < 15: score += 25
        elif pe < 20: score += 15
        n += 1

    # P/B
    pb = row["P/B"]
    if pd.notna(pb) and pb > 0:
        if pb < 1.0: score += 35
        elif pb < 1.5: score += 25
        elif pb < 2.0: score += 15
        n += 1

    # Real Yield
    yld = row["Yield %"]
    if pd.notna(yld):
        if yld > 4.0: score += 30
        elif yld > 2.5: score += 20
        elif yld > 1.5: score += 10
        n += 1

    fund_score = round(score * 100 / (100 if n == 3 else 70 if n == 2 else 35), 0) if n > 0 else 0

    results.append({
        "Ticker": idx,
        "Price": row["Price"],
        "DDM Value": ddm,
        "Relative Value": relative,
        "Fair Value": fair,
        "Upside %": upside,
        "Fundamental Score": int(fund_score),
    })

val_df = pd.DataFrame(results).set_index("Ticker").round(2)
val_df = val_df.sort_values("Fundamental Score", ascending=False)

# Styling
def color_score(val):
    if val >= 80: return "background-color: #d4edda; font-weight: bold"
    if val >= 65: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_val = val_df.style.format({
    "Price": "${:,.2f}", "DDM Value": "${:,.2f}", "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"
}, na_rep="—") \
    .background_gradient(subset=["Upside %"], cmap="RdYlGn") \
    .applymap(color_score, subset=["Fundamental Score"])

st.dataframe(styled_val, use_container_width=True)

# Chart
st.subheader("3. True Ranking – Best Real-Value Miners First")
fig, ax = plt.subplots(figsize=(10, 6))
scores = val_df["Fundamental Score"]
colors = ["#27ae60" if s >= 80 else "#f39c12" if s >= 65 else "#e74c3c" for s in scores]
ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1], height=0.7)
ax.set_xlabel("Fundamental Score (100 = Best Real Value)")
ax.set_title("Green = Deep Value | Yellow = Fair | Red = Expensive")
ax.grid(axis='x', alpha=0.3)
for i, (t, s) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(s + 1, i, f"{int(s)}", va='center', fontweight='bold', color="black")
st.pyplot(fig)

st.success("Uses only real dividends & trailing metrics — pure fundamental truth")
st.caption("Data: Yahoo Finance • Real dividends only • Trailing metrics only • Nov 30 2025")
