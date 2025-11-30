import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Mining Screener", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Miners – Smart Peer-Relative Scoring")
st.markdown("**P/E • P/B • ROE • Profit Margin → True 0–100 Score (Higher = Better)**")

# -------------------------------
# STOCK INPUT
# -------------------------------
all_miners = ["AEM","NEM","GOLD","WPM","FNV","PAAS","KGC","SSRM","RGLD","HL","CDE","SAND","OR","EXK","AGI","BTG","HMY"]

manual_input = st.text_input(
    "Enter tickers (space or comma separated):",
    placeholder="e.g. AEM PAAS WPM GOLD KGC"
)

if manual_input.strip():
    tickers = [t.strip().upper() for t in manual_input.replace(",", " ").split() if t.strip()]
else:
    default = ["AEM", "WPM", "FNV", "PAAS", "KGC", "GOLD", "RGLD", "SSRM"]
    tickers = st.multiselect("Or select from list:", all_miners, default=default)

if not tickers:
    st.info("Enter or select tickers to begin")
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
        except:
            data.append({"Ticker": t, "Name": "Error", "Price": None, "RSI (14)": None, "Yield %": 0,
                         "Annual Div $": 0, "P/E": None, "P/B": None, "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# -------------------------------
# YOUR EXACT SCORING FUNCTION
# -------------------------------
def calculate_fundamental_score(row, valid_pe, valid_pb, valid_roe, valid_margin):
    score_components = []

    # P/E (lower better)
    if row["P/E"] is not None and row["P/E"] > 0 and len(valid_pe) > 1:
        pe_score = 100 * (max(valid_pe) - row["P/E"]) / (max(valid_pe) - min(valid_pe))
        score_components.append((pe_score, 0.30))

    # P/B (lower better)
    if row["P/B"] is not None and row["P/B"] > 0 and len(valid_pb) > 1:
        pb_score = 100 * (max(valid_pb) - row["P/B"]) / (max(valid_pb) - min(valid_pb))
        score_components.append((pb_score, 0.25))

    # ROE (higher better)
    if row["ROE %"] is not None and len(valid_roe) > 1:
        roe_score = 100 * (row["ROE %"] - min(valid_roe)) / (max(valid_roe) - min(valid_roe))
        score_components.append((roe_score, 0.25))

    # Profit Margin (higher better)
    if row["Profit Margin %"] is not None and len(valid_margin) > 1:
        margin_score = 100 * (row["Profit Margin %"] - min(valid_margin)) / (max(valid_margin) - min(valid_margin))
        score_components.append((margin_score, 0.20))

    if not score_components:
        return None

    total_weight = sum(w for _, w in score_components)
    final_score = sum(s * w for s, w in score_components) / total_weight
    return round(final_score, 1)

# Extract valid values for normalization
valid_pe = [v for v in df["P/E"] if v is not None and v > 0]
valid_pb = [v for v in df["P/B"] if v is not None and v > 0]
valid_roe = [v for v in df["ROE %"] if v is not None]
valid_margin = [v for v in df["Profit Margin %"] if v is not None]

# Apply scoring
df["Fundamental Score"] = df.apply(
    lambda row: calculate_fundamental_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1
)

# -------------------------------
# 1. VALUATION TABLE (P/E + P/B Relative)
# -------------------------------
st.subheader("1. Relative Fair Value (Trailing P/E + P/B)")

peer_pe = df["P/E"].median()
peer_pb = df["P/B"].median()

val_rows = []
for idx, row in df.iterrows():
    rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else None
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else None
    fair = np.nanmean([rel_pe, rel_pb]) if rel_pe or rel_pb else None
    upside = (fair / row["Price"] - 1) * 100 if fair and row["Price"] else None

    val_rows.append({
        "Ticker": idx,
        "Price": row["Price"],
        "P/E Fair": rel_pe,
        "P/B Fair": rel_pb,
        "Fair Value": fair,
        "Upside %": upside,
    })

val_df = pd.DataFrame(val_rows).round(2).set_index("Ticker")
st.dataframe(
    val_df.style.format({
        "Price": "${:,.2f}", "P/E Fair": "${:,.2f}", "P/B Fair": "${:,.2f}",
        "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"
    }, na_rep="—").background_gradient(subset=["Upside %"], cmap="RdYlGn"),
    use_container_width=True
)

# -------------------------------
# 2. RSI + YIELD + SMART SCORE TABLE
# -------------------------------
st.subheader("2. Technical + Income + Smart Fundamental Score")

score_table = df[["RSI (14)", "Yield %", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]].copy()
score_table = score_table.round(2)

def color_score(val):
    if pd.isna(val) or val is None: return ""
    if val >= 80: return "background-color: #d4edda; font-weight: bold"
    if val >= 65: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

styled_score = score_table.style.format({
    "RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "P/E": "{:.1f}", "P/B": "{:.2f}",
    "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"
}, na_rep="—").applymap(color_score, subset=["Fundamental Score"])

st.dataframe(styled_score, use_container_width=True)

# -------------------------------
# 3. RANKING CHART USING YOUR SCORE
# -------------------------------
st.subheader("3. Best Opportunities – Ranked by Smart Score")

scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)

if not scored.empty:
    scores = scored["Fundamental Score"]
    colors = ["#27ae60" if s >= 80 else "#f39c12" if s >= 65 else "#e74c3c" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(scored)))
    ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100)")
    ax.set_title("Green = Top Tier Value • Yellow = Good • Red = Avoid")
    ax.grid(axis='x', alpha=0.3)
    for i, (ticker, s) in enumerate(zip(scored.index[::-1], scores[::-1])):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold', fontsize=10)
    st.pyplot(fig)
else:
    st.warning("Not enough data to calculate scores")

st.success("Your exact scoring logic implemented • Peer-relative • Robust • Beautiful")
st.caption("Data: Yahoo Finance • Real-time • Not financial advice • 2025")
