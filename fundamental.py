import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Value Screener", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Miners – Pure Value Screener")
st.markdown("**No DDM • No forward estimates • Only trailing P/E, P/B & real dividends**")

# -------------------------------
# STOCK SELECTION AT THE TOP
# -------------------------------
all_miners = {
    "Producers": ["AEM", "NEM", "GOLD", "KGC", "PAAS", "SSR", "HL", "CDE", "EXK"],
    "Royalty/Streaming": ["WPM", "FNV", "RGLD", "OR", "SAND", "AGI"],
    "Large Cap": ["NEM", "AEM", "GOLD", "WPM", "FNV"],
    "Mid/Small Cap": ["PAAS", "KGC", "SSRM", "HL", "CDE", "SAND"]
}

col1, col2 = st.columns([1, 3])
with col1:
    preset = st.selectbox("Quick List:", list(all_miners.keys()))
with col2:
    selected_tickers = st.multiselect(
        "Or pick your own (overrides list):",
        options=sorted([t for group in all_miners.values() for t in group]),
        default=all_miners[preset][:8]
    )

tickers = selected_tickers or all_miners[preset]
st.markdown(f"**Selected ({len(tickers)} stocks):** {', '.join(tickers)}")

# -------------------------------
# FETCH REAL TRAILING DATA
# -------------------------------
@st.cache_data(ttl=3600)
def get_trailing_data(tickers):
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan

            # Last real dividend (for display only)
            divs = stock.dividends
            recent = divs[divs.index > divs.index.max() - pd.Timedelta(days=400)]
            last_div = recent.iloc[-1] if not recent.empty else 0
            annual_div = last_div * 4 if last_div > 0 else 0
            yield_pct = (annual_div / price) * 100 if price and annual_div > 0 else 0

            rows.append({
                "Ticker": t,
                "Name": info.get("longName", t),
                "Price": price,
                "Last Qtr Div $": round(last_div, 4),
                "Annual Div $": round(annual_div, 3),
                "Yield %": round(yield_pct, 2),
                "Trailing P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Market Cap $B": round(info.get("marketCap", 0) / 1e9, 2),
            })
        except:
            rows.append({k: np.nan for k in ["Ticker","Name","Price","Last Qtr Div $","Annual Div $","Yield %","Trailing P/E","P/B","ROE %","Market Cap $B"]})
            rows[-1]["Ticker"] = t
    return pd.DataFrame(rows)

df = get_trailing_data(tickers)

# -------------------------------
# RELATIVE VALUATION ONLY
# -------------------------------
st.subheader("Relative Fair Value (Trailing P/E + P/B)")

peer_pe = df["Trailing P/E"].median(skipna=True)
peer_pb = df["P/B"].median(skipna=True)

results = []
for _, row in df.iterrows():
    # Relative from P/E
    rel_pe = row["Price"] * (peer_pe / row["Trailing P/E"]) if pd.notna(row["Trailing P/E"]) and row["Trailing P/E"] > 0 else np.nan
    # Relative from P/B
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    # Final Relative Value
    fair_value = np.nanmean([rel_pe, rel_pb])

    upside = (fair_value / row["Price"] - 1) * 100 if pd.notna(fair_value) and pd.notna(row["Price"]) else np.nan

    # Fundamental Score (0–100) – trailing only
    score = 0
    n = 0
    pe = row["Trailing P/E"]
    if pd.notna(pe) and pe > 0:
        if pe < 10: score += 40
        elif pe < 15: score += 30
        elif pe < 20: score += 15
        n += 1
    pb = row["P/B"]
    if pd.notna(pb) and pb > 0:
        if pb < 1.0: score += 40
        elif pb < 1.5: score += 30
        elif pb < 2.0: score += 15
        n += 1
    yld = row["Yield %"]
    if pd.notna(yld):
        if yld > 4: score += 20
        elif yld > 2.5: score += 15
        elif yld > 1.5: score += 10
        n += 1

    fund_score = round(score * 100 / (100 if n == 3 else 80 if n == 2 else 40), 0) if n > 0 else 0

    results.append({
        "Ticker": row["Ticker"],
        "Price": row["Price"],
        "P/E Fair": rel_pe,
        "P/B Fair": rel_pb,
        "Fair Value": fair_value,
        "Upside %": upside,
        "Score": int(fund_score),
    })

val_df = pd.DataFrame(results).round(2)
val_df = val_df.sort_values("Score", ascending=False).set_index("Ticker")

# Styling
def color_score(val):
    if val >= 80: return "background-color: #d4edda; font-weight: bold"
    if val >= 65: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled = val_df.style.format({
    "Price": "${:,.2f}",
    "P/E Fair": "${:,.2f}",
    "P/B Fair": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%"
}, na_rep="—") \
    .background_gradient(subset=["Upside %"], cmap="RdYlGn") \
    .applymap(color_score, subset=["Score"])

st.dataframe(styled, use_container_width=True)

# Ranking Chart
st.subheader("Value Ranking – Best Deals First")
fig, ax = plt.subplots(figsize=(10, 6))
scores = val_df["Score"]
colors = ["#27ae60" if s >= 80 else "#f39c12" if s >= 65 else "#e74c3c" for s in scores]
ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1], height=0.7)
ax.set_xlabel("Fundamental Score (100 = Cheapest & Strongest)")
ax.set_title("Green = Deep Value • Yellow = Fair • Red = Expensive")
ax.grid(axis='x', alpha=0.3)
for i, (t, s) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(s + 1, i, f"{int(s)}", va='center', fontweight='bold')
st.pyplot(fig)

# Summary
st.markdown("### How Relative Value Works")
st.info("""
**Fair Value = Average of:**
- **P/E Method**: Price × (Peer Median P/E ÷ This Stock’s P/E)
- **P/B Method**: Price × (Peer Median P/B ÷ This Stock’s P/B)

→ If a stock has lower P/E or P/B than peers → **undervalued** → higher fair value
""")

st.success("Pure trailing value screener • No DDM • No forward guesses • Just real numbers")
st.caption("Data: Yahoo Finance • Real dividends shown for reference only • Not financial advice • 2025")
