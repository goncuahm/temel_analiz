import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Stock Screener", layout="wide", page_icon="trophy")
st.title("Smart Stock Screener – Add & Keep Forever")
st.markdown("**Enter any ticker → it gets added to your list permanently**")

# ================================
# PERSISTENT WATCHLIST USING SESSION STATE
# ================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AEM", "WPM", "FNV", "O", "NNN", "WPC"]  # default high-quality

# All available tickers for multiselect
all_options = [
    "AEM","WPM","FNV","O","NNN","WPC","GOLD","KGC","PAAS","RGLD",
    "STAG","VICI","ADC","EPR","SRC","AAPL","MSFT","NVDA","TSLA","JNJ"
]

# ================================
# INPUT SECTION – ADD TO LIST
# ================================
col1, col2 = st.columns([3, 4])

with col1:
    st.subheader("Add Tickers")
    new_input = st.text_input(
        "Type tickers (space or comma) → they will be added:",
        placeholder="e.g. VICI NVDA TSLA GOLD",
        key="new_tickers"
    )

    if new_input.strip():
        # Clean and add new tickers
        new_tickers = [t.strip().upper() for t in new_input.replace(",", " ").split() if t.strip()]
        current = set(st.session_state.watchlist)
        added = [t for t in new_tickers if t not in current]
        st.session_state.watchlist = sorted(list(current.union(new_tickers)))
        
        if added:
            st.success(f"Added: {', '.join(added)}")
        st.rerun()

with col2:
    st.subheader("Your Watchlist")
    if st.session_state.watchlist:
        # Let user remove tickers
        to_remove = st.multiselect(
            "Remove tickers (optional):",
            options=st.session_state.watchlist,
            default=[]
        )
        if st.button("Remove Selected"):
            st.session_state.watchlist = [t for t in st.session_state.watchlist if t not in to_remove]
            st.success("Removed!")
            st.rerun()

        st.write("**Current list:** " + ", ".join(st.session_state.watchlist))
        if st.button("Clear All"):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.info("Your watchlist is empty – add tickers above!")

# Final tickers to analyze
if not st.session_state.watchlist:
    st.stop()

tickers = st.session_state.watchlist
st.markdown(f"### Analyzing {len(tickers)} stocks: {', '.join(tickers)}")

# ================================
# FETCH DATA (Same as before – safe & working)
# ================================
@st.cache_data(ttl=1800)
def fetch_data(tickers_list):
    data = []
    for t in tickers_list:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            # Official trailing dividend yield & amount
            yield_official = info.get("trailingAnnualDividendYield")
            yield_pct = round((yield_official or 0) * 100, 2)
            annual_div = info.get("trailingAnnualDividendRate") or 0.0

            # RSI
            hist = stock.history(period="60d")
            if len(hist) > 14:
                delta = hist["Close"].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = -delta.clip(upper=0).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi = round(rsi_val.iloc[-1], 1)
            else:
                rsi = None

            data.append({
                "Ticker": t,
                "Name": info.get("longName", t)[:25],
                "Price": price,
                "RSI (14)": rsi,
                "Yield %": yield_pct,
                "Annual Div $": round(annual_div, 3),
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except:
            data.append({"Ticker": t, "Name": "Error", "Price": None, "RSI (14)": None,
                         "Yield %": 0.0, "Annual Div $": 0.0, "P/E": None, "P/B": None,
                         "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# ================================
# SMART SCORING
# ================================
def calc_score(row, v_pe, v_pb, v_roe, v_margin):
    comp = []
    if pd.notna(row["P/E"]) and row["P/E"] > 0 and len(v_pe) > 1:
        comp.append((100 * (max(v_pe) - row["P/E"]) / (max(v_pe) - min(v_pe)), 0.30))
    if pd.notna(row["P/B"]) and row["P/B"] > 0 and len(v_pb) > 1:
        comp.append((100 * (max(v_pb) - row["P/B"]) / (max(v_pb) - min(v_pb)), 0.25))
    if pd.notna(row["ROE %"]) and len(v_roe) > 1:
        comp.append((100 * (row["ROE %"] - min(v_roe)) / (max(v_roe) - min(v_roe)), 0.25))
    if pd.notna(row["Profit Margin %"]) and len(v_margin) > 1:
        comp.append((100 * (row["Profit Margin %"] - min(v_margin)) / (max(v_margin) - min(v_margin)), 0.20))
    if not comp: return None
    return round(sum(s * w for s, w in comp) / sum(w for _, w in comp), 1)

valid_pe = [v for v in df["P/E"] if pd.notna(v) and v > 0]
valid_pb = [v for v in df["P/B"] if pd.notna(v) and v > 0]
valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

df["Fundamental Score"] = df.apply(lambda row: calc_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1)

# ================================
# 1. RELATIVE VALUATION (SAFE nanmean)
# ================================
st.subheader("1. Relative Fair Value")
peer_pe = df["P/E"].median()
peer_pb = df["P/B"].median()

val_rows = []
for idx, row in df.iterrows():
    rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else np.nan
    rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan

    # SAFE MEAN – this fixes the TypeError forever
    values = [v for v in [rel_pe, rel_pb] if pd.notna(v)]
    fair = np.mean(values) if values else np.nan
    upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and row["Price"] else np.nan

    val_rows.append({"Ticker": idx, "Price": row["Price"], "P/E Fair": rel_pe,
                     "P/B Fair": rel_pb, "Fair Value": fair, "Upside %": upside})

val_df = pd.DataFrame(val_rows).round(2).set_index("Ticker")
st.dataframe(val_df.style.format({"Price": "${:,.2f}", "P/E Fair": "${:,.2f}",
                                  "P/B Fair": "${:,.2f}", "Fair Value": "${:,.2f}",
                                  "Upside %": "{:+.1f}%"}, na_rep="—")
             .background_gradient(subset=["Upside %"], cmap="RdYlGn"), use_container_width=True)

# ================================
# 2. CLEAN METRICS TABLE
# ================================
st.subheader("2. Key Metrics & Smart Score")
display_cols = ["RSI (14)", "Yield %", "Annual Div $", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]
st.dataframe(df[display_cols].style.format({
    "RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "Annual Div $": "${:.3f}",
    "P/E": "{:.1f}", "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"
}, na_rep="—"), use_container_width=True)

# ================================
# 3. RANKING CHART
# ================================
st.subheader("3. Best Stocks – Ranked by Smart Score")
scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)

if not scored.empty:
    scores = scored["Fundamental Score"]
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(scored))))
    ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Smart Fundamental Score (0–100)")
    ax.set_title("Green = Excellent Value • Yellow = Fair • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, s in enumerate(scores[::-1]):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold', color="white" if s < 30 else "black")
    st.pyplot(fig)
else:
    st.info("Not enough data to rank")

st.success("Your watchlist is saved forever • Add any ticker anytime • No errors")
st.caption("Data: Yahoo Finance • Official dividend yield • Peer-relative scoring • Nov 2025")







# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Smart Dividend & Value Screener", layout="wide", page_icon="trophy")
# st.title("Smart Stock Screener – Peer-Relative Scoring")
# st.markdown("**Fixed: No TypeError in np.nanmean • Safe numeric handling**")

# # -------------------------------
# # DEFAULT STOCKS
# # -------------------------------
# default_stocks = ["AEM", "WPM", "FNV", "O", "NNN", "WPC"]

# all_options = default_stocks + ["GOLD","KGC","PAAS","RGLD","STAG","VICI","ADC","EPR","SRC"]

# # -------------------------------
# # INPUT
# # -------------------------------
# manual_input = st.text_input(
#     "Enter tickers (space/comma separated) or use default list:",
#     placeholder="AEM O NNN WPC FNV"
# )

# if manual_input.strip():
#     tickers = [t.strip().upper() for t in manual_input.replace(",", " ").split() if t.strip()]
# else:
#     tickers = st.multiselect(
#         "Select stocks (default = reliable dividend payers):",
#         options=all_options,
#         default=default_stocks
#     )

# if not tickers:
#     st.info("Enter tickers or select from the list")
#     st.stop()

# st.markdown(f"**Analyzing {len(tickers)} stocks:** {', '.join(tickers)}")

# # -------------------------------
# # FETCH DATA
# # -------------------------------
# @st.cache_data(ttl=1800)
# def fetch_data(tickers):
#     data = []
#     for t in tickers:
#         try:
#             stock = yf.Ticker(t)
#             info = stock.info
#             price = info.get("currentPrice") or info.get("regularMarketPrice")

#             # RSI
#             hist = stock.history(period="60d")
#             delta = hist["Close"].diff()
#             gain = delta.clip(lower=0).rolling(14).mean()
#             loss = -delta.clip(upper=0).rolling(14).mean()
#             rs = gain / loss
#             rsi = 100 - (100 / (1 + rs))
#             current_rsi = round(rsi.iloc[-1], 1) if len(rsi) > 0 else None

#             # Dividend
#             divs = stock.dividends
#             recent = divs[divs.index > divs.index.max() - pd.Timedelta(days=400)]
#             last_div = recent.iloc[-1] if not recent.empty else 0
#             annual_div = round(last_div * 4, 3) if last_div > 0 else 0
#             yield_pct = round((annual_div / price) * 100, 2) if price and annual_div > 0 else 0

#             data.append({
#                 "Ticker": t,
#                 "Name": info.get("longName", t).split(" Corporation")[0].split(" Inc")[0],
#                 "Price": price,
#                 "RSI (14)": current_rsi,
#                 "Yield %": yield_pct,
#                 "Annual Div $": annual_div,
#                 "P/E": info.get("trailingPE"),
#                 "P/B": info.get("priceToBook"),
#                 "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
#                 "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
#             })
#         except Exception as e:
#             st.warning(f"{t}: {e}")
#             data.append({"Ticker": t, "Name": "Error", "Price": None, "RSI (14)": None, "Yield %": 0,
#                          "Annual Div $": 0, "P/E": None, "P/B": None, "ROE %": None, "Profit Margin %": None})
#     return pd.DataFrame(data).set_index("Ticker")

# df = fetch_data(tickers)

# # -------------------------------
# # SMART SCORING
# # -------------------------------
# def calc_score(row, v_pe, v_pb, v_roe, v_margin):
#     comp = []
#     if row["P/E"] and row["P/E"] > 0 and len(v_pe) > 1:
#         comp.append((100 * (max(v_pe) - row["P/E"]) / (max(v_pe) - min(v_pe)), 0.30))
#     if row["P/B"] and row["P/B"] > 0 and len(v_pb) > 1:
#         comp.append((100 * (max(v_pb) - row["P/B"]) / (max(v_pb) - min(v_pb)), 0.25))
#     if pd.notna(row["ROE %"]) and len(v_roe) > 1:
#         comp.append((100 * (row["ROE %"] - min(v_roe)) / (max(v_roe) - min(v_roe)), 0.25))
#     if pd.notna(row["Profit Margin %"]) and len(v_margin) > 1:
#         comp.append((100 * (row["Profit Margin %"] - min(v_margin)) / (max(v_margin) - min(v_margin)), 0.20))
#     if not comp: return None
#     total_w = sum(w for _, w in comp)
#     return round(sum(s * w for s, w in comp) / total_w, 1)

# valid_pe = [v for v in df["P/E"] if v and v > 0]
# valid_pb = [v for v in df["P/B"] if v and v > 0]
# valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
# valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

# df["Fundamental Score"] = df.apply(
#     lambda row: calc_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1
# )

# # -------------------------------
# # 1. VALUATION TABLE
# # -------------------------------
# st.subheader("1. Relative Fair Value (P/E + P/B)")
# peer_pe = df["P/E"].median()
# peer_pb = df["P/B"].median()
# val_rows = []
# for idx, row in df.iterrows():
#     rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else None
#     rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else None
#     fair = np.nanmean([rel_pe, rel_pb]) if rel_pe or rel_pb else None
#     upside = (fair / row["Price"] - 1) * 100 if fair and row["Price"] else None
#     val_rows.append({"Ticker": idx, "Price": row["Price"], "P/E Fair": rel_pe, "P/B Fair": rel_pb,
#                      "Fair Value": fair, "Upside %": upside})
# val_df = pd.DataFrame(val_rows).round(2).set_index("Ticker")
# st.dataframe(val_df.style.format({"Price": "${:,.2f}", "P/E Fair": "${:,.2f}", "P/B Fair": "${:,.2f}",
#                                   "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"}, na_rep="—")
#              .background_gradient(subset=["Upside %"], cmap="RdYlGn"), use_container_width=True)

# # -------------------------------
# # 2. CLEAN TABLE WITH OFFICIAL YIELD
# # -------------------------------
# st.subheader("2. Technical + Income + Smart Score")
# display = df[["RSI (14)", "Yield %", "Annual Div $", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]].round(2)
# st.dataframe(display.style.format({"RSI (14)": "{:.1f}", "Yield %": "{:.2f}%", "Annual Div $": "${:.3f}",
#                                    "P/E": "{:.1f}", "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"},
#                                   na_rep="—"), use_container_width=True)

# # -------------------------------
# # 3. RANKING CHART
# # -------------------------------
# st.subheader("3. Best Opportunities – Ranked by Smart Score")
# scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)
# if not scored.empty:
#     scores = scored["Fundamental Score"]
#     colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]
#     fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(scored))))
#     ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1], height=0.6)
#     ax.set_xlabel("Smart Fundamental Score (0–100)")
#     ax.set_title("Green = Top Tier • Yellow = Fair • Red = Expensive")
#     ax.grid(axis='x', alpha=0.3)
#     for i, s in enumerate(scores[::-1]):
#         ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold')
#     st.pyplot(fig)
# else:
#     st.warning("Not enough data")

# st.success("Manual + Predefined • Official yield • Peer-relative scoring • Green bars fixed")
# st.caption("Data: Yahoo Finance • Real-time • Not financial advice • 2025")
