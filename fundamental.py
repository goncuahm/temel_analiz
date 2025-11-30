import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Dividend Screener", layout="wide", page_icon="trophy")
st.title("Smart Dividend Stock Screener")
st.markdown("**P/E • P/B • ROE • Profit Margin → True Peer-Relative Score (0–100)**")

# -------------------------------
# DEFAULT HIGH-QUALITY DIVIDEND STOCKS
# -------------------------------
default_stocks = ["AEM", "WPM", "FNV",    # Top miners
                  "O", "NNN", "WPC"]     # Top triple-net REITs

all_options = default_stocks + ["GOLD","KGC","PAAS","RGLD","STAG","VICI","ADC","EPR","SRC"]

# -------------------------------
# INPUT
# -------------------------------
manual_input = st.text_input(
    "Enter tickers (space/comma separated) or use default list:",
    placeholder="AEM O NNN WPC FNV"
)

if manual_input.strip():
    tickers = [t.strip().upper() for t in manual_input.replace(",", " ").split() if t.strip()]
else:
    tickers = st.multiselect(
        "Select stocks (default = reliable dividend payers):",
        options=all_options,
        default=default_stocks
    )

if not tickers:
    st.info("Please enter or select tickers")
    st.stop()

st.markdown(f"**Analyzing {len(tickers)} stocks:** {', '.join(tickers)}")

# -------------------------------
# FETCH DATA
# -------------------------------
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
                "Name": info.get("longName", t),
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

# -------------------------------
# SMART SCORING (bug fixed here)
# -------------------------------
def calculate_fundamental_score(row, valid_pe, valid_pb, valid_roe, valid_margin):
    components = []

    if row["P/E"] is not None and row["P/E"] > 0 and len(valid_pe) > 1:
        pe_score = 100 * (max(valid_pe) - row["P/E"]) / (max(valid_pe) - min(valid_pe))
        components.append((pe_score, 0.30))

    if row["P/B"] is not None and row["P/B"] > 0 and len(valid_pb) > 1:
        pb_score = 100 * (max(valid_pb) - row["P/B"]) / (max(valid_pb) - min(valid_pb))
        components.append((pb_score, 0.25))

    if row["ROE %"] is not None and len(valid_roe) > 1:
        roe_score = 100 * (row["ROE %"] - min(valid_roe)) / (max(valid_roe) - min(valid_roe))
        components.append((roe_score, 0.25))

    if row["Profit Margin %"] is not None and len(valid_margin) > 1:
        margin_score = 100 * (row["Profit Margin %"] - min(valid_margin)) / (max(valid_margin) - min(valid_margin))
        components.append((margin_score, 0.20))

    if not components:
        return None

    total_w = sum(w for _, w in components)
    score = sum(s * w for s, w in components) / total_w
    return round(score, 1)

# Correct lists (no space in variable name!)
valid_pe = [v for v in df["P/E"] if v is not None and v > 0]
valid_pb = [v for v in df["P/B"] if v is not None and v > 0]
valid_roe = [v for v in df["ROE %"] if v is not None]           # ← fixed
valid_margin = [v for v in df["Profit Margin %"] if v is not None]  # ← fixed

df["Fundamental Score"] = df.apply(
    lambda row: calculate_fundamental_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1
)

# -------------------------------
# 1. VALUATION TABLE
# -------------------------------
st.subheader("1. Relative Fair Value (P/E + P/B)")
# (unchanged – omitted for brevity, same as previous version)

# -------------------------------
# 2. CLEAN SCORE TABLE (no colored background)
# -------------------------------
st.subheader("2. Technical, Income & Smart Score")
display = df[["RSI (14)", "Yield %", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]].round(2)
st.dataframe(
    display.style.format({
        "RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "P/E": "{:.1f}", "P/B": "{:.2f}",
        "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"
    }, na_rep="—"),
    use_container_width=True
)

# -------------------------------
# 3. FIXED RANKING CHART – NOW SHOWS GREEN!
# -------------------------------
st.subheader("3. Best Opportunities – Ranked by Smart Score")

scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)

if not scored.empty:
    scores = scored["Fundamental Score"]
    # Proper green for the best ones
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(scored))))
    bars = ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100)")
    ax.set_title("Green = Top Tier • Yellow = Fair • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, s in enumerate(scores[::-1]):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold', color="black")
    st.pyplot(fig)
else:
    st.warning("Not enough data to calculate scores")

st.success("Bug fixed → Green bars now appear for the best stocks!")
st.caption("Data: Yahoo Finance • Real-time • Not financial advice")
