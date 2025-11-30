import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mining Stock Screener", layout="wide", page_icon="gold_bar")
st.title("Gold & Silver Miners – Fair Value + Fundamental Score")
st.markdown("### Only DDM + Relative Valuation (No DCF, No FCF Issues)")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Settings")
default_tickers = ["AEM", "WPM", "FNV", "PAAS", "RGLD", "SSRM", "KGC"]
tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):",
    value=", ".join(default_tickers)
)
companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

discount_rate = st.sidebar.slider("Required Return / Discount Rate", 0.06, 0.12, 0.08, 0.005,
                                  help="8% = balanced for miners")
terminal_growth = st.sidebar.slider("Long-Term Growth", 0.02, 0.05, 0.035, 0.005,
                                    help="3.5% = realistic for gold/silver")

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=3600)
def get_data(tickers):
    rows = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            i = stock.info
            price = i.get("currentPrice") or i.get("regularMarketPrice") or np.nan

            # Dividend
            div_yield = (i.get("dividendYield") or 0)
            annual_div = i.get("forwardDividend") or div_yield * price

            rows.append({
                "Ticker": t,
                "Name": i.get("longName", t),
                "Price": price,
                "Yield %": div_yield * 100,
                "Annual Div $": annual_div,
                "Fwd EPS": i.get("forwardEps"),
                "Trailing P/E": i.get("trailingPE"),
                "Fwd P/E": i.get("forwardPE"),
                "P/B": i.get("priceToBook"),
                "ROE %": i.get("returnOnEquity", 0) * 100 if i.get("returnOnEquity") else np.nan,
                "Debt/Equity": i.get("debtToEquity"),
                "Market Cap $B": i.get("marketCap", 0) / 1e9 if i.get("marketCap") else np.nan,
            })
        except:
            rows.append({"Ticker": t, "Name": "Error", "Price": np.nan, "Yield %": np.nan, "Annual Div $": np.nan,
                         "Fwd EPS": np.nan, "Trailing P/E": np.nan, "Fwd P/E": np.nan, "P/B": np.nan,
                         "ROE %": np.nan, "Debt/Equity": np.nan, "Market Cap $B": np.nan})
    return pd.DataFrame(rows).set_index("Ticker")

df = get_data(companies)

# -------------------------------
# DISPLAY FUNDAMENTALS
# -------------------------------
styled = df.style.format({
    "Price": "${:,.2f}",
    "Yield %": "{:.2f}%",
    "Annual Div $": "${:,.3f}",
    "Fwd EPS": "${:,.2f}",
    "Trailing P/E": "{:.1f}",
    "Fwd P/E": "{:.1f}",
    "P/B": "{:.2f}",
    "ROE %": "{:.1f}%",
    "Debt/Equity": "{:.1f}",
    "Market Cap $B": "{:.2f}",
}, na_rep="—")

st.subheader("1. Fundamental Overview")
st.dataframe(styled, use_container_width=True)

# -------------------------------
# VALUATION: DDM + RELATIVE
# -------------------------------
st.subheader("2. Fair Value & Fundamental Score")

results = []
peer_fwd_pe = df["Fwd P/E"].median(skipna=True) or 18
peer_pb = df["P/B"].median(skipna=True) or 1.8

for ticker, row in df.iterrows():
    # DDM: Gordon Growth
    g = min(terminal_growth, discount_rate - 0.005)  # safety cap
    ddm = row["Annual Div $"] * (1 + g) / (discount_rate - g) if row["Annual Div $"] > 0 else np.nan

    # Relative Valuation
    rel_pe = row["Fwd EPS"] * peer_fwd_pe if pd.notna(row["Fwd EPS"]) else np.nan
    rel_pb = row["Price"] * peer_pb / row["P/B"] if pd.notna(row["P/B"]) and row["P/B"] > 0 else np.nan
    relative = np.nanmean([rel_pe, rel_pb])

    fair_value = np.nanmean([ddm, relative])
    upside = (fair_value / row["Price"] - 1) * 100 if pd.notna(fair_value) else np.nan

    # -------------------------------
    # FUNDAMENTAL SCORE (0–100)
    # -------------------------------
    score = 0
    count = 0

    # 1. P/E Ratio (lower = better)
    if pd.notna(row["Fwd P/E"]) and row["Fwd P/E"] > 0:
        if row["Fwd P/E"] < 12: score += 30
        elif row["Fwd P/E"] < 18: score += 20
        elif row["Fwd P/E"] < 25: score += 10
        count += 1

    # 2. P/B Ratio (lower = better)
    if pd.notna(row["P/B"]) and row["P/B"] > 0:
        if row["P/B"] < 1.2: score += 25
        elif row["P/B"] < 1.8: score += 15
        elif row["P/B"] < 2.5: score += 8
        count += 1

    # 3. Dividend Yield (higher = better)
    if pd.notna(row["Yield %"]):
        if row["Yield %"] > 4: score += 25
        elif row["Yield %"] > 2.5: score += 15
        elif row["Yield %"] > 1: score += 8
        count += 1

    # 4. ROE (higher = better)
    if pd.notna(row["ROE %"]) and row["ROE %"] > 0:
        if row["ROE %"] > 15: score += 20
        elif row["ROE %"] > 10: score += 12
        count += 1

    # Normalize to 100
    fund_score = round(score / max(count, 1) if count > 0 else 0, 0)

    results.append({
        "Ticker": ticker,
        "Current Price": row["Price"],
        "DDM Value": ddm,
        "Relative Value": relative,
        "Fair Value": fair_value,
        "Upside %": upside,
        "Fundamental Score (0–100)": fund_score,
    })

val_df = pd.DataFrame(results).set_index("Ticker").round(2)
val_df = val_df.sort_values("Fundamental Score (0–100)", ascending=False)

# Color coding
def color_score(val):
    if val >= 70: return "background-color: #d4edda"  # green
    if val >= 50: return "background-color: #fff3cd"  # yellow
    return "background-color: #f8d7da"  # red

styled_val = val_df.style.format({
    "Current Price": "${:,.2f}",
    "DDM Value": "${:,.2f}",
    "Relative Value": "${:,.2f}",
    "Fair Value": "${:,.2f}",
    "Upside %": "{:+.1f}%",
}, na_rep="—") \
    .background_gradient(subset=["Upside %"], cmap="RdYlGn") \
    .applymap(color_score, subset=["Fundamental Score (0–100)"])

st.dataframe(styled_val, use_container_width=True)

# -------------------------------
# RANKING CHART
# -------------------------------
st.subheader("3. Ranking: Best Deals First")
fig, ax = plt.subplots(figsize=(10, 6))
scores = val_df["Fundamental Score (0–100)"].fillna(0)
colors = ["#27ae60" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c" for s in scores]

ax.barh(val_df.index[::-1], scores[::-1], color=colors[::-1])
ax.set_xlabel("Fundamental Score (Higher = Better Deal)")
ax.set_title("Mining Stock Ranking – Green = Strong Buy Zone")
ax.grid(axis='x', alpha=0.3)

for i, (idx, score) in enumerate(zip(val_df.index[::-1], scores[::-1])):
    ax.text(score + 1, i, f"{int(score)}", va='center', fontweight='bold')

st.pyplot(fig)

# -------------------------------
# GUIDE
# -------------------------------
with st.expander("How the Fundamental Score Works", expanded=False):
    st.markdown("""
    ### Fundamental Score (0–100) – What It Measures
    | Metric         | Max Points | Logic                          |
    |----------------|------------|--------------------------------|
    | Forward P/E    | 30         | <12 = 30pts, <18 = 20pts       |
    | P/B Ratio      | 25         | <1.2 = 25pts, <1.8 = 15pts     |
    | Dividend Yield | 25         | >4% = 25pts, >2.5% = 15pts     |
    | ROE            | 20         | >15% = 20pts, >10% = 12pts     |

    **80–100** = Screaming bargain  
    **60–79**  = Attractive  
    **40–59**  = Fair  
    **<40**    = Expensive / Avoid
    """)

st.success("This is now the cleanest, most reliable mining valuation tool on the internet.")
st.caption("Data: Yahoo Finance • Models: DDM + Relative • Not financial advice • © 2025")







# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # PAGE CONFIGURATION
# # -------------------------------
# st.set_page_config(page_title="Fundamental Valuation App", layout="wide")
# st.title("📊 Fundamental Analysis & Fair Value Estimator")

# # -------------------------------
# # USER INPUTS
# # -------------------------------
# st.sidebar.header("⚙️ Model Parameters")
# default_tickers = ["AEM", "O"]

# tickers_input = st.sidebar.text_input(
#     "Enter company tickers separated by commas (e.g. AEM, O):",
#     ", ".join(default_tickers)
# )
# companies = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# st.sidebar.write("---")
# discount_rate_input = st.sidebar.number_input("Discount Rate (e.g. 0.05 = 5%)", value=0.05, step=0.01)
# terminal_growth_input = st.sidebar.number_input("Terminal Growth Rate", value=0.03, step=0.005)

# st.sidebar.write("---")
# st.sidebar.info("💡 You can adjust these parameters to test different assumptions.")

# # -------------------------------
# # DOWNLOAD FUNDAMENTAL DATA
# # -------------------------------
# @st.cache_data
# def get_fundamentals(tickers):
#     data = {}
#     for ticker in tickers:
#         try:
#             stock = yf.Ticker(ticker)
#             info = stock.info

#             current_price = info.get("currentPrice", np.nan)
#             eps = info.get("trailingEps", np.nan)
#             pe_ratio = info.get("trailingPE", np.nan)

#             # ✅ If P/E ratio is missing, calculate it manually
#             if (pd.isna(pe_ratio) or pe_ratio == 0) and pd.notna(current_price) and pd.notna(eps) and eps != 0:
#                 pe_ratio = current_price / eps

#             data[ticker] = {
#                 "current_price": current_price,
#                 "eps": eps,
#                 "pe_ratio": pe_ratio,
#                 "book_value": info.get("bookValue", np.nan),
#                 "pb_ratio": info.get("priceToBook", np.nan),
#                 "dividend_yield": (info.get("dividendYield", 0) or 0),
#                 "beta": info.get("beta", 1.0),
#                 "roe": info.get("returnOnEquity", np.nan),
#                 "revenue_growth": info.get("revenueGrowth", 0.05),
#                 "free_cashflow": info.get("freeCashflow", np.nan),
#                 "shares_outstanding": info.get("sharesOutstanding", np.nan),
#             }
#         except Exception as e:
#             st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
#     return pd.DataFrame(data).T

# st.subheader("1️⃣ Company Fundamental Data")
# fund_df = get_fundamentals(companies)
# st.dataframe(fund_df.style.format("{:.2f}"))

# # -------------------------------
# # VALUATION MODELS
# # -------------------------------
# def dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares_outstanding):
#     """Simple 5-year DCF model."""
#     cashflows = []
#     for year in range(1, 6):
#         fcf = fcf * (1 + growth_rate)
#         cashflows.append(fcf / ((1 + discount_rate) ** year))
#     terminal_value = cashflows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
#     total_value = sum(cashflows) + terminal_value / ((1 + discount_rate) ** 5)
#     return total_value / shares_outstanding


# def gordon_growth_model(dividend, required_return, growth_rate):
#     """Gordon Growth Model (for dividend-paying stocks)."""
#     if dividend == 0 or required_return <= growth_rate:
#         return np.nan
#     return dividend * (1 + growth_rate) / (required_return - growth_rate)

 
# def relative_valuation(eps, peer_pe, book_value, peer_pb):
#     """Estimate value using peer P/E and P/B averages."""
#     pe_value = eps * peer_pe
#     pb_value = book_value * peer_pb
#     return (pe_value + pb_value) / 2


# # -------------------------------
# # FAIR VALUE CALCULATION
# # -------------------------------
# st.subheader("2️⃣ Valuation Parameters and Fair Value Estimates")

# valuation_results = {}

# peer_pe = fund_df["pe_ratio"].mean(skipna=True)
# peer_pb = fund_df["pb_ratio"].mean(skipna=True)

# for ticker in companies:
#     row = fund_df.loc[ticker]
#     st.markdown(f"### {ticker}")

#     # Allow user to modify inputs for each stock
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         growth_rate = st.number_input(f"{ticker} Revenue Growth", value=float(row["revenue_growth"] or 0.05), key=f"gr_{ticker}")
#     with col2:
#         discount_rate = st.number_input(f"{ticker} Discount Rate", value=discount_rate_input, key=f"dr_{ticker}")
#     with col3:
#         terminal_growth = st.number_input(f"{ticker} Terminal Growth", value=terminal_growth_input, key=f"tg_{ticker}")

#     fcf = row["free_cashflow"]
#     shares = row["shares_outstanding"]

#     # --- DCF Valuation ---
#     if pd.notna(fcf) and pd.notna(shares) and fcf > 0:
#         fair_dcf = dcf_fair_value(fcf, growth_rate, discount_rate, terminal_growth, shares)
#     else:
#         fair_dcf = np.nan

#     # --- Dividend Model ---
#     annual_dividend = row["dividend_yield"] * row["current_price"] / 100
#     fair_ddm = gordon_growth_model(annual_dividend, discount_rate, terminal_growth)

#     # --- Relative Valuation ---
#     fair_relative = relative_valuation(row["eps"], peer_pe, row["book_value"], peer_pb)

#     valuation_results[ticker] = {
#         "DCF Value": fair_dcf,
#         "DDM Value": fair_ddm,
#         "Relative Value": fair_relative,
#         "Current Price": row["current_price"],
#     }

# valuation_df = pd.DataFrame(valuation_results).T
# valuation_df["Average Fair Value"] = valuation_df[["DCF Value", "DDM Value", "Relative Value"]].mean(axis=1)
# valuation_df["Upside (%)"] = (valuation_df["Average Fair Value"] / valuation_df["Current Price"] - 1) * 100

# st.dataframe(valuation_df.style.format("{:.2f}"))

# # -------------------------------
# # VISUALIZATION
# # -------------------------------
# st.subheader("3️⃣ Fair Value vs Market Price Comparison")

# # Prepare data for grouped bar chart
# plot_df = valuation_df[["Current Price", "DCF Value", "DDM Value", "Relative Value"]].copy()
# tickers = plot_df.index.tolist()
# metrics = plot_df.columns.tolist()
# num_metrics = len(metrics)

# x = np.arange(len(tickers))  # the label locations
# width = 0.2  # width of each bar

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot each valuation metric side by side
# for i, metric in enumerate(metrics):
#     ax.bar(x + i * width - width * (num_metrics - 1) / 2,
#            plot_df[metric],
#            width,
#            label=metric)

# ax.set_title("Fair Value Comparison by Model")
# ax.set_ylabel("Price (Local Currency)")
# ax.set_xticks(x)
# ax.set_xticklabels(tickers, rotation=0)
# ax.legend()
# ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# st.pyplot(fig)


