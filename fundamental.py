import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Dividend & Value Screener", layout="wide", page_icon="trophy")
st.title("Smart Stock Screener – Peer-Relative Scoring")
st.markdown("**Now with RSI (9-day) • 4 Top Mining + 5 REITs Default**")

# -------------------------------
# DEFAULT: 4 Mining + 5 REITs (US)
# -------------------------------
default_stocks = ["AEM", "WPM", "FNV", "NEM",    # Top Mining / Royalty
                  "O", "NNN", "WPC", "ADC", "VICI"]  # Top Dividend REITs

all_options = default_stocks + ["GOLD","RGLD","SAND","STAG","EPR","SRC","OR","PAAS"]

# -------------------------------
# INPUT
# -------------------------------
manual_input = st.text_input(
    "Enter tickers (space/comma separated) or use default list:",
    placeholder="e.g. EREGL.IS PAGYO.IS THYAO.IS"
)

if manual_input.strip():
    tickers = [t.strip().upper() for t in manual_input.replace(",", " ").split() if t.strip()]
else:
    tickers = st.multiselect(
        "Or select from defaults (4 Mining + 5 REITs):",
        options=all_options,
        default=default_stocks
    )

if not tickers:
    st.info("Enter tickers or select from the list")
    st.stop()

st.success(f"Analyzing {len(tickers)} stock(s): {', '.join(tickers)}")

# -------------------------------
# FETCH DATA – RSI (9-DAY) + OFFICIAL YIELD
# -------------------------------
@st.cache_data(ttl=1800)
def fetch_data(tickers):
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

            # === RSI (9-day) ===
            hist = stock.history(period="60d")
            rsi_val = None
            if len(hist) >= 20:  # need at least 20 days for reliable 9-day RSI
                delta = hist["Close"].diff()
                gain = delta.clip(lower=0).rolling(window=9).mean()
                loss = -delta.clip(upper=0).rolling(window=9).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs))
                current_rsi = round(rsi_val.iloc[-1], 1)
            else:
                current_rsi = None

            # Official dividend yield (trailing 12 months)
            yield_pct = round((info.get("trailingAnnualDividendYield") or 0) * 100, 2)
            annual_div = round(info.get("trailingAnnualDividendRate") or 0, 3)

            data.append({
                "Ticker": t,
                "Name": info.get("longName", t)[:30],
                "Price": price,
                "RSI (9)": current_rsi,           # Now 9-day
                "Yield %": yield_pct,
                "Annual Div $": annual_div,
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except Exception as e:
            st.warning(f"{t}: Data error")
            data.append({"Ticker": t, "Name": "N/A", "Price": None, "RSI (9)": None,
                         "Yield %": 0, "Annual Div $": 0, "P/E": None, "P/B": None,
                         "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# -------------------------------
# SMART SCORING
# -------------------------------
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
    total_w = sum(w for _, w in comp)
    return round(sum(s * w for s, w in comp) / total_w, 1)

valid_pe = [v for v in df["P/E"] if pd.notna(v) and v > 0]
valid_pb = [v for v in df["P/B"] if pd.notna(v) and v > 0]
valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

df["Fundamental Score"] = df.apply(lambda row: calc_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1)

# -------------------------------
# 1. RELATIVE FAIR VALUE
# -------------------------------
st.subheader("1. Relative Fair Value")
peer_pe = df["P/E"].median()
peer_pb = df["P/B"].median()
rows = []
for idx, row in df.iterrows():
    rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else np.nan
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    fair = np.nanmean([rel_pe, rel_pb]) if pd.notna(rel_pe) or pd.notna(rel_pb) else np.nan
    upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and pd.notna(row["Price"]) and row["Price"] > 0 else np.nan
    rows.append({"Ticker": idx, "Price": row["Price"], "Fair Value": fair, "Upside %": upside})

val_df = pd.DataFrame(rows).round(2).set_index("Ticker")
st.dataframe(val_df.style.format({"Price": "${:,.2f}", "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"}, na_rep="—")
             .background_gradient(subset=["Upside %"], cmap="RdYlGn"), use_container_width=True)

# -------------------------------
# 2. MAIN TABLE – NOW WITH RSI (9)
# -------------------------------
st.subheader("2. Key Metrics & Smart Score")
display_cols = ["RSI (9)", "Yield %", "Annual Div $", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]
st.dataframe(df[display_cols].style.format({
    "RSI (9)": "{:.1f}",
    "Yield %": "{:.2f}%",
    "Annual Div $": "${:.3f}",
    "P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "ROE %": "{:.1f}%",
    "Profit Margin %": "{:.1f}%"
}, na_rep="—"), use_container_width=True)

# -------------------------------
# 3. RANKING CHART
# -------------------------------
st.subheader("3. Best Opportunities – Ranked by Smart Score")
scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)
if not scored.empty:
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scored["Fundamental Score"]]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(scored))))
    ax.barh(scored.index[::-1], scored["Fundamental Score"][::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100)")
    ax.set_title("Green = Excellent Value • Yellow = Fair • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, s in enumerate(scored["Fundamental Score"][::-1]):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold', color="white" if s < 40 else "black")
    st.pyplot(fig)
else:
    st.warning("Not enough data for scoring")

st.success("RSI (9-day) active • 4 Mining + 5 REITs default • Works globally including .IS")
st.caption("Data: Yahoo Finance • Real-time • RSI 9-day • Not financial advice • 2025")
