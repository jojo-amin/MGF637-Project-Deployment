import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(page_title="Undervalued Stock Screener", layout="wide")

st.title("Undervalued Stock Screener")
st.write(
    "This app screens stocks using valuation, momentum, and risk metrics. "
    "It ranks companies based on a weighted attractiveness score."
)

default_tickers = "AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, JPM, BAC, XOM, CVX, JNJ, PFE, KO, WMT, COST"

sector_map = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Cyclical",
    "META": "Communication Services",
    "NVDA": "Technology",
    "AMD": "Technology",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "KO": "Consumer Defensive",
    "WMT": "Consumer Defensive",
    "COST": "Consumer Defensive",
}

tickers_input = st.text_area(
    "Enter stock tickers separated by commas:",
    value=default_tickers
)

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

st.sidebar.header("Scoring Weights")

value_weight = st.sidebar.slider("Value Weight", 0.0, 1.0, 0.4, 0.05)
momentum_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.35, 0.05)
risk_weight = st.sidebar.slider("Risk Weight", 0.0, 1.0, 0.25, 0.05)

total_weight = value_weight + momentum_weight + risk_weight

if total_weight == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

value_weight = value_weight / total_weight
momentum_weight = momentum_weight / total_weight
risk_weight = risk_weight / total_weight


def clean_number(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def get_quote_data(tickers):
    symbols = ",".join(tickers)
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"

    rows = []

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        results = data.get("quoteResponse", {}).get("result", [])

        quote_dict = {item.get("symbol"): item for item in results}

        for ticker in tickers:
            item = quote_dict.get(ticker, {})

            rows.append({
                "Ticker": ticker,
                "Company": item.get("shortName", ticker),
                "Sector": sector_map.get(ticker, "Unknown"),
                "Market Cap": clean_number(item.get("marketCap", np.nan)),
                "Price": clean_number(item.get("regularMarketPrice", np.nan)),
                "P/E": clean_number(item.get("trailingPE", np.nan)),
                "Forward P/E": clean_number(item.get("forwardPE", np.nan)),
                "P/B": clean_number(item.get("priceToBook", np.nan)),
            })

    except Exception:
        for ticker in tickers:
            rows.append({
                "Ticker": ticker,
                "Company": ticker,
                "Sector": sector_map.get(ticker, "Unknown"),
                "Market Cap": np.nan,
                "Price": np.nan,
                "P/E": np.nan,
                "Forward P/E": np.nan,
                "P/B": np.nan,
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_price_metrics(tickers):
    try:
        price_data = yf.download(
            tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True
        )

        rows = []

        if len(tickers) == 1:
            close_prices = price_data["Close"].to_frame(name=tickers[0])
        else:
            close_prices = price_data["Close"]

        for ticker in tickers:
            try:
                prices = close_prices[ticker].dropna()

                if len(prices) < 2:
                    one_year_return = np.nan
                    volatility = np.nan
                    risk_adjusted_return = np.nan
                    max_drawdown = np.nan
                    latest_price = np.nan
                else:
                    latest_price = prices.iloc[-1]
                    first_price = prices.iloc[0]

                    one_year_return = (latest_price - first_price) / first_price

                    daily_returns = prices.pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252)

                    if volatility == 0 or pd.isna(volatility):
                        risk_adjusted_return = np.nan
                    else:
                        risk_adjusted_return = one_year_return / volatility

                    rolling_max = prices.cummax()
                    drawdowns = (prices - rolling_max) / rolling_max
                    max_drawdown = drawdowns.min()

                rows.append({
                    "Ticker": ticker,
                    "Price from History": latest_price,
                    "One Year Return": one_year_return,
                    "Annualized Volatility": volatility,
                    "Risk-Adjusted Return": risk_adjusted_return,
                    "Max Drawdown": max_drawdown
                })

            except Exception:
                rows.append({
                    "Ticker": ticker,
                    "Price from History": np.nan,
                    "One Year Return": np.nan,
                    "Annualized Volatility": np.nan,
                    "Risk-Adjusted Return": np.nan,
                    "Max Drawdown": np.nan
                })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame({
            "Ticker": tickers,
            "Price from History": np.nan,
            "One Year Return": np.nan,
            "Annualized Volatility": np.nan,
            "Risk-Adjusted Return": np.nan,
            "Max Drawdown": np.nan
        })


def percentile_score(series, higher_is_better=True):
    clean_series = pd.to_numeric(series, errors="coerce")
    clean_series = clean_series.replace([np.inf, -np.inf], np.nan)

    if clean_series.notna().sum() == 0:
        return pd.Series([50] * len(clean_series), index=clean_series.index)

    if higher_is_better:
        return clean_series.rank(pct=True) * 100
    else:
        return (1 - clean_series.rank(pct=True)) * 100


def calculate_scores(df):
    scored = df.copy()

    numeric_cols = [
        "P/E",
        "Forward P/E",
        "P/B",
        "One Year Return",
        "Annualized Volatility",
        "Risk-Adjusted Return",
        "Max Drawdown"
    ]

    for col in numeric_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    scored.loc[scored["P/E"] <= 0, "P/E"] = np.nan
    scored.loc[scored["Forward P/E"] <= 0, "Forward P/E"] = np.nan
    scored.loc[scored["P/B"] <= 0, "P/B"] = np.nan

    scored["P/E Score"] = percentile_score(scored["P/E"], higher_is_better=False)
    scored["Forward P/E Score"] = percentile_score(scored["Forward P/E"], higher_is_better=False)
    scored["P/B Score"] = percentile_score(scored["P/B"], higher_is_better=False)

    scored["Value Score"] = scored[
        ["P/E Score", "Forward P/E Score", "P/B Score"]
    ].mean(axis=1)

    scored["Momentum Score"] = percentile_score(scored["One Year Return"], higher_is_better=True)

    scored["Risk-Adjusted Score"] = percentile_score(scored["Risk-Adjusted Return"], higher_is_better=True)
    scored["Volatility Score"] = percentile_score(scored["Annualized Volatility"], higher_is_better=False)
    scored["Drawdown Score"] = percentile_score(scored["Max Drawdown"], higher_is_better=True)

    scored["Risk Score"] = scored[
        ["Risk-Adjusted Score", "Volatility Score", "Drawdown Score"]
    ].mean(axis=1)

    scored["Final Score"] = (
        value_weight * scored["Value Score"] +
        momentum_weight * scored["Momentum Score"] +
        risk_weight * scored["Risk Score"]
    )

    scored["Final Score"] = scored["Final Score"].fillna(0)

    return scored.sort_values("Final Score", ascending=False)


def format_large_number(x):
    if pd.isna(x):
        return "N/A"
    if x >= 1_000_000_000_000:
        return f"${x / 1_000_000_000_000:.2f}T"
    if x >= 1_000_000_000:
        return f"${x / 1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"${x / 1_000_000:.2f}M"
    return f"${x:,.0f}"


if st.button("Run Stock Screener"):
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    else:
        with st.spinner("Pulling stock data and calculating scores..."):
            quote_df = get_quote_data(tickers)
            price_df = get_price_metrics(tickers)

            df = pd.merge(quote_df, price_df, on="Ticker", how="left")

            df["Price"] = df["Price"].fillna(df["Price from History"])

            scored_df = calculate_scores(df)

        st.success("Stock screener completed.")

        st.subheader("Ranked Stock Screener Results")

        display_cols = [
            "Ticker",
            "Company",
            "Sector",
            "Market Cap",
            "Price",
            "P/E",
            "Forward P/E",
            "P/B",
            "One Year Return",
            "Annualized Volatility",
            "Risk-Adjusted Return",
            "Max Drawdown",
            "Value Score",
            "Momentum Score",
            "Risk Score",
            "Final Score"
        ]

        st.dataframe(
            scored_df[display_cols],
            use_container_width=True
        )

        top_stock = scored_df.iloc[0]

        st.subheader("Top Ranked Stock")

        st.write(
            f"**{top_stock['Ticker']}** ranks highest based on the selected scoring weights."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Top Stock", top_stock["Ticker"])

        with col2:
            st.metric(
                "Final Score",
                f"{top_stock['Final Score']:.2f}"
            )

        with col3:
            st.metric(
                "1-Year Return",
                "N/A" if pd.isna(top_stock["One Year Return"]) else f"{top_stock['One Year Return'] * 100:.2f}%"
            )

        with col4:
            st.metric(
                "Market Cap",
                format_large_number(top_stock["Market Cap"])
            )

        st.subheader("Final Score by Stock")

        fig_score = px.bar(
            scored_df,
            x="Ticker",
            y="Final Score",
            color="Sector",
            title="Stock Attractiveness Score"
        )

        st.plotly_chart(fig_score, use_container_width=True)

        st.subheader("Value Score vs Momentum Score")

        scatter_df = scored_df.copy()
        scatter_df["Market Cap"] = pd.to_numeric(scatter_df["Market Cap"], errors="coerce")

        market_cap_median = scatter_df["Market Cap"].median()

        if pd.isna(market_cap_median):
            scatter_df["Market Cap Size"] = 1
        else:
            scatter_df["Market Cap Size"] = scatter_df["Market Cap"].fillna(market_cap_median)

        fig_scatter = px.scatter(
            scatter_df,
            x="Value Score",
            y="Momentum Score",
            size="Market Cap Size",
            color="Sector",
            hover_name="Ticker",
            title="Value Score vs Momentum Score"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("One-Year Return by Stock")

        return_df = scored_df.copy()
        return_df["One Year Return Percent"] = return_df["One Year Return"] * 100

        fig_return = px.bar(
            return_df,
            x="Ticker",
            y="One Year Return Percent",
            color="Sector",
            title="One-Year Stock Return (%)"
        )

        st.plotly_chart(fig_return, use_container_width=True)

        st.subheader("Risk Comparison")

        risk_df = scored_df.copy()
        risk_df["Annualized Volatility Percent"] = risk_df["Annualized Volatility"] * 100
        risk_df["Max Drawdown Percent"] = risk_df["Max Drawdown"] * 100

        fig_vol = px.bar(
            risk_df,
            x="Ticker",
            y="Annualized Volatility Percent",
            color="Sector",
            title="Annualized Volatility (%)"
        )

        st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Valuation Multiples")

        valuation_df = scored_df[["Ticker", "P/E", "Forward P/E", "P/B"]].copy()
        valuation_df = valuation_df.set_index("Ticker")

        fig_val = px.bar(
            valuation_df,
            barmode="group",
            title="Comparison of Valuation Multiples"
        )

        st.plotly_chart(fig_val, use_container_width=True)

        st.subheader("How to Interpret the Results")

        st.write(
            "A higher final score means the stock ranked better relative to the other stocks in the sample. "
            "The model rewards cheaper valuation multiples, stronger recent momentum, and lower risk."
        )

        st.write(
            "This does not mean the top-ranked stock is guaranteed to be a good investment. "
            "The screener is best used as a first-pass filtering tool before deeper financial analysis."
        )

        st.warning(
            "Note: Some fundamental Yahoo Finance fields may be unavailable on Streamlit Cloud. "
            "The app handles this by still ranking stocks using available valuation, momentum, and risk data."
        )

        csv = scored_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="stock_screener_results.csv",
            mime="text/csv"
        )

else:
    st.info("Enter tickers and click **Run Stock Screener** to begin.")
