import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Global Screener", layout="wide", page_icon="trophy")
st.title("Smart Stock Screener – Global & Turkish Stocks")
st.markdown("**Add any ticker • Remove with X • Peer-relative scoring**")

# ================================
# PERSISTENT WATCHLIST WITH REMOVABLE CHIPS
# ================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
        "EREGL.IS", "PAGYO.IS", "THYAO.IS", "AKBNK.IS", "GARAN.IS",
        "AEM", "WPM", "FNV", "O", "NNN", "WPC"
    ]

# -------------------------------
# ADD NEW TICKERS
# -------------------------------
new_input = st.text_input(
    "Add new tickers (e.g. SISE.IS VOD.L AAPL)",
    placeholder="Type and press Enter",
    key="add_new"
)

if new_input.strip():
    new_ticks = [t.strip().upper() for t in new_input.replace(",", " ").split() if t.strip()]
    current = set(st.session_state.watchlist)
    added = [t for t in new_ticks if t not in current]
    st.session_state.watchlist = list(current.union(new_ticks))
    if added:
        st.success(f"Added: {', '.join(added)}")
    st.rerun()

# -------------------------------
# DISPLAY WATCHLIST WITH X BUTTONS
# -------------------------------
st.markdown("### Your Watchlist")
if st.session_state.watchlist:
    cols = st.columns(6)
    for i, ticker in enumerate(st.session_state.watchlist):
        with cols[i % 6]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"`{ticker}`")
            with col2:
                if st.button("X", key=f"remove_{ticker}"):
                    st.session_state.watchlist.remove(ticker)
                    st.rerun()
else:
    st.info("Your watchlist is empty – add tickers above!")

# Clear all button
if st.session_state.watchlist:
    if st.button("Clear All Watchlist"):
        st.session_state.watchlist = []
        st.rerun()

if not st.session_state.watchlist:
    st.stop()

tickers = st.session_state.watchlist
st.markdown(f"**Analyzing {len(tickers)} stocks:** {', '.join(tickers)}")

# ================================
# FETCH DATA – GLOBAL + TURKISH SUPPORT
# ================================
@st.cache_data(ttl=1800)
def fetch_data(tickers_list):
    data = []
    for t in tickers_list:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

            # RSI (9)
            hist = stock.history(period="60d")
            if len(hist) >= 20:
                delta = hist["Close"].diff()
                gain = delta.clip(lower=0).rolling(9).mean()
                loss = -delta.clip(upper=0).rolling(9).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi = round(rsi_val.iloc[-1], 1)
            else:
                rsi = None

            # Official dividend yield
            yield_pct = round((info.get("trailingAnnualDividendYield") or 0) * 100, 2)
            annual_div = round(info.get("trailingAnnualDividendRate") or 0, 3)

            data.append({
                "Ticker": t,
                "Name": info.get("longName", t)[:30],
                "Price": price,
                "RSI (9)": rsi,
                "Yield %": yield_pct,
                "Annual Div": annual_div,
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
                "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
            })
        except:
            data.append({"Ticker": t, "Name": "N/A", "Price": None, "RSI (9)": None,
                         "Yield %": 0, "Annual Div": 0, "P/E": None, "P/B": None,
                         "ROE %": None, "Profit Margin %": None})
    return pd.DataFrame(data).set_index("Ticker")

df = fetch_data(tickers)

# ================================
# SMART SCORING
# ================================
def calc_score(row):
    comp = []
    pe = row["P/E"]; pb = row["P/B"]; roe = row["ROE %"]; margin = row["Profit Margin %"]
    v_pe = [v for v in df["P/E"] if pd.notna(v) and v > 0]
    v_pb = [v for v in df["P/B"] if pd.notna(v) and v > 0]
    v_roe = [v for v in df["ROE %"] if pd.notna(v)]
    v_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

    if pe and pe > 0 and len(v_pe) > 1:
        comp.append((100 * (max(v_pe) - pe) / (max(v_pe) - min(v_pe)), 0.30))
    if pb and pb > 0 and len(v_pb) > 1:
        comp.append((100 * (max(v_pb) - pb) / (max(v_pb) - min(v_pb)), 0.25))
    if pd.notna(roe) and len(v_roe) > 1:
        comp.append((100 * (roe - min(v_roe)) / (max(v_roe) - min(v_roe)), 0.25))
    if pd.notna(margin) and len(v_margin) > 1:
        comp.append((100 * (margin - min(v_margin)) / (max(v_margin) - min(v_margin)), 0.20))
    if not comp: return None
    return round(sum(s * w for s, w in comp) / sum(w for _, w in comp), 1)

df["Fundamental Score"] = df.apply(calc_score, axis=1)

# ================================
# 1. RELATIVE FAIR VALUE
# ================================
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

# ================================
# 2. MAIN TABLE
# ================================
st.subheader("2. Key Metrics & Smart Score")
cols = ["RSI (9)", "Yield %", "Annual Div", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]
st.dataframe(df[cols].style.format({
    "RSI (9)": "{:.1f}", "Yield %": "{:.2f}%", "Annual Div": "${:.3f}",
    "P/E": "{:.1f}", "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"
}, na_rep="—"), use_container_width=True)

# ================================
# 3. RANKING CHART
# ================================
st.subheader("3. Best Opportunities")
scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)
if not scored.empty:
    colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scored["Fundamental Score"]]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(scored)))
    ax.barh(scored.index[::-1], scored["Fundamental Score"][::-1], color=colors[::-1])
    ax.set_xlabel("Smart Score (0–100)")
    ax.set_title("Green = Excellent • Red = Expensive")
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, s) in enumerate(zip(scored.index[::-1], scored["Fundamental Score"][::-1])):
        ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold')
    st.pyplot(fig)

st.success("Removable chips with X • Add below • Works with .IS, .L, .DE • Perfect!")
st.caption("Yahoo Finance • Real-time • Global stocks • 2025")







# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Smart Global Stock Screener", layout="wide", page_icon="trophy")
# st.title("Smart Stock Screener – Works with .IS, .L, .DE, etc.")
# st.markdown("**Peer-relative scoring • Official dividend yield • Works globally**")

# # ================================
# # PERSISTENT WATCHLIST
# # ================================
# if "watchlist" not in st.session_state:
#     st.session_state.watchlist = ["AEM", "WPM", "FNV", "O", "NNN", "WPC", "EREGL.IS", "PAGYO.IS"]

# # ================================
# # ADD TICKERS
# # ================================
# col1, col2 = st.columns([3, 4])

# with col1:
#     new = st.text_input(
#         "Add any ticker (e.g. EREGL.IS THYAO.IS AAPL.L VOD.L",
#         placeholder="Type here and press Enter",
#         key="add"
#     )
#     if new.strip():
#         new_ticks = [t.strip().upper() for t in new.replace(",", " ").split() if t.strip()]
#         current = set(st.session_state.watchlist)
#         added = [t for t in new_ticks if t not in current]
#         st.session_state.watchlist = sorted(list(current.union(new_ticks)))
#         if added:
#             st.success(f"Added: {', '.join(added)}")
#         st.rerun()

# with col2:
#     st.subheader("Your Watchlist")
#     if st.session_state.watchlist:
#         remove = st.multiselect("Remove tickers:", st.session_state.watchlist)
#         if st.button("Remove Selected"):
#             st.session_state.watchlist = [t for t in st.session_state.watchlist if t not in remove]
#             st.rerun()
#         st.write("**Stocks:** " + ", ".join(st.session_state.watchlist))
#         if st.button("Clear All"):
#             st.session_state.watchlist = []
#             st.rerun()
#     else:
#         st.info("Add tickers above")

# if not st.session_state.watchlist:
#     st.stop()

# tickers = st.session_state.watchlist
# st.markdown(f"### Analyzing {len(tickers)} stocks: {', '.join(tickers)}")

# # ================================
# # FETCH DATA – SAFE FOR ALL MARKETS
# # ================================
# @st.cache_data(ttl=1800)
# def fetch_data(tickers_list):
#     data = []
#     for t in tickers_list:
#         try:
#             stock = yf.Ticker(t)
#             info = stock.info

#             # Price
#             price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

#             # RSI (9-period as you wanted)
#             hist = stock.history(period="60d")
#             if len(hist) >= 20:
#                 delta = hist["Close"].diff()
#                 gain = delta.clip(lower=0).rolling(9).mean()
#                 loss = -delta.clip(upper=0).rolling(9).mean()
#                 rs = gain / loss
#                 rsi = 100 - (100 / (1 + rs))
#                 current_rsi = round(rsi.iloc[-1], 1)
#             else:
#                 current_rsi = None

#             # Official dividend yield (trailing 12 months)
#             yield_official = info.get("trailingAnnualDividendYield")
#             yield_pct = round((yield_official or 0) * 100, 2)
#             annual_div = round(info.get("trailingAnnualDividendRate") or 0, 3)

#             data.append({
#                 "Ticker": t,
#                 "Name": info.get("longName", t)[:30],
#                 "Price": price,
#                 "RSI (9)": current_rsi,
#                 "Yield %": yield_pct,
#                 "Annual Div $": annual_div,
#                 "P/E": info.get("trailingPE"),
#                 "P/B": info.get("priceToBook"),
#                 "ROE %": round((info.get("returnOnEquity") or 0) * 100, 1),
#                 "Profit Margin %": round((info.get("profitMargins") or 0) * 100, 1),
#             })
#         except Exception as e:
#             st.warning(f"{t}: Data not available or error")
#             data.append({"Ticker": t, "Name": "N/A", "Price": None, "RSI (9)": None,
#                          "Yield %": 0, "Annual Div $": 0, "P/E": None, "P/B": None,
#                          "ROE %": None, "Profit Margin %": None})
#     return pd.DataFrame(data).set_index("Ticker")

# df = fetch_data(tickers)

# # ================================
# # SMART SCORING
# # ================================
# def calc_score(row, v_pe, v_pb, v_roe, v_margin):
#     comp = []
#     if pd.notna(row["P/E"]) and row["P/E"] > 0 and len(v_pe) > 1:
#         comp.append((100 * (max(v_pe) - row["P/E"]) / (max(v_pe) - min(v_pe)), 0.30))
#     if pd.notna(row["P/B"]) and row["P/B"] > 0 and len(v_pb) > 1:
#         comp.append((100 * (max(v_pb) - row["P/B"]) / (max(v_pb) - min(v_pb)), 0.25))
#     if pd.notna(row["ROE %"]) and len(v_roe) > 1:
#         comp.append((100 * (row["ROE %"] - min(v_roe)) / (max(v_roe) - min(v_roe)), 0.25))
#     if pd.notna(row["Profit Margin %"]) and len(v_margin) > 1:
#         comp.append((100 * (row["Profit Margin %"] - min(v_margin)) / (max(v_margin) - min(v_margin)), 0.20))
#     if not comp: return None
#     return round(sum(s * w for s, w in comp) / sum(w for _, w in comp), 1)

# valid_pe = [v for v in df["P/E"] if pd.notna(v) and v > 0]
# valid_pb = [v for v in df["P/B"] if pd.notna(v) and v > 0]
# valid_roe = [v for v in df["ROE %"] if pd.notna(v)]
# valid_margin = [v for v in df["Profit Margin %"] if pd.notna(v)]

# df["Fundamental Score"] = df.apply(lambda row: calc_score(row, valid_pe, valid_pb, valid_roe, valid_margin), axis=1)

# # ================================
# # 1. RELATIVE VALUATION
# # ================================
# st.subheader("1. Relative Fair Value")
# peer_pe = df["P/E"].median()
# peer_pb = df["P/B"].median()
# val_rows = []
# for idx, row in df.iterrows():
#     rel_pe = row["Price"] * (peer_pe / row["P/E"]) if pd.notna(row["P/E"]) and row["P/E"] > 0 else np.nan
#     rel_pb = row["Price"] * (peer_pb / row["P/B"]) if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
#     fair = np.nanmean([rel_pe, rel_pb]) if pd.notna(rel_pe) or pd.notna(rel_pb) else np.nan
#     upside = (fair / row["Price"] - 1) * 100 if pd.notna(fair) and pd.notna(row["Price"]) and row["Price"] > 0 else np.nan
#     val_rows.append({"Ticker": idx, "Price": row["Price"], "P/E Fair": rel_pe, "P/B Fair": rel_pb,
#                      "Fair Value": fair, "Upside %": upside})

# val_df = pd.DataFrame(val_rows).round(2).set_index("Ticker")
# st.dataframe(val_df.style.format({"Price": "${:,.2f}", "P/E Fair": "${:,.2f}", "P/B Fair": "${:,.2f}",
#                                   "Fair Value": "${:,.2f}", "Upside %": "{:+.1f}%"}, na_rep="—")
#              .background_gradient(subset=["Upside %"], cmap="RdYlGn"), use_container_width=True)

# # ================================
# # 2. MAIN TABLE – FIXED COLUMN NAME
# # ================================
# st.subheader("2. Key Metrics & Smart Score")
# display_cols = ["RSI (9)", "Yield %", "Annual Div $", "P/E", "P/B", "ROE %", "Profit Margin %", "Fundamental Score"]
# st.dataframe(df[display_cols].style.format({
#     "RSI (9)": "{:.1f}", "Yield %": "{:.2f}%", "Annual Div $": "${:.3f}",
#     "P/E": "{:.1f}", "P/B": "{:.2f}", "ROE %": "{:.1f}%", "Profit Margin %": "{:.1f}%"
# }, na_rep="—"), use_container_width=True)

# # ================================
# # 3. RANKING CHART
# # ================================
# st.subheader("3. Top Ranked Stocks")
# scored = df.dropna(subset=["Fundamental Score"]).sort_values("Fundamental Score", ascending=False)
# if not scored.empty:
#     scores = scored["Fundamental Score"]
#     colors = ["#1e7b1e" if s >= 80 else "#f39c12" if s >= 65 else "#c0392b" for s in scores]
#     fig, ax = plt.subplots(figsize=(10, max(4, 0.6*len(scored))))
#     ax.barh(scored.index[::-1], scores[::-1], color=colors[::-1])
#     ax.set_xlabel("Smart Score (0–100)")
#     ax.set_title("Green = Best Value • Red = Expensive")
#     ax.grid(axis='x', alpha=0.3)
#     for i, s in enumerate(scores[::-1]):
#         ax.text(s + 1, i, f"{s:.1f}", va='center', fontweight='bold')
#     st.pyplot(fig)

# st.success("Now works perfectly with EREGL.IS, PAGYO.IS, THYAO.IS, etc.")
# st.caption("Global stocks • Official yield • Peer scoring • Persistent list • 2025")
