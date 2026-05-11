import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Undervalued Stock Screener", layout="wide")

st.title("Undervalued Stock Screener")
st.write(
    "This app screens stocks using valuation, quality, and momentum metrics. "
    "It ranks companies based on a weighted attractiveness score."
)

default_tickers = "AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, JPM, BAC, XOM, CVX, JNJ, PFE, KO, WMT, COST"

tickers_input = st.text_area(
    "Enter stock tickers separated by commas:",
    value=default_tickers
)

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

st.sidebar.header("Scoring Weights")

value_weight = st.sidebar.slider("Value Weight", 0.0, 1.0, 0.4, 0.05)
quality_weight = st.sidebar.slider("Quality Weight", 0.0, 1.0, 0.4, 0.05)
momentum_weight = st.sidebar.slider("Momentum Weight", 0.0, 1.0, 0.2, 0.05)

total_weight = value_weight + quality_weight + momentum_weight

if total_weight == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

value_weight = value_weight / total_weight
quality_weight = quality_weight / total_weight
momentum_weight = momentum_weight / total_weight


def clean_number(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


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
                    latest_price = np.nan
                else:
                    latest_price = prices.iloc[-1]
                    first_price = prices.iloc[0]
                    one_year_return = (latest_price - first_price) / first_price
                    daily_returns = prices.pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252)

                rows.append({
                    "Ticker": ticker,
                    "Price": latest_price,
                    "One Year Return": one_year_return,
                    "Annualized Volatility": volatility
                })

            except Exception:
                rows.append({
                    "Ticker": ticker,
                    "Price": np.nan,
                    "One Year Return": np.nan,
                    "Annualized Volatility": np.nan
                })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame({
            "Ticker": tickers,
            "Price": np.nan,
            "One Year Return": np.nan,
            "Annualized Volatility": np.nan
        })


@st.cache_data(show_spinner=False)
def get_fundamental_metrics(tickers):
    rows = []

    for ticker in tickers:
        company = np.nan
        sector = "Unknown"
        market_cap = np.nan
        pe = np.nan
        forward_pe = np.nan
        ps = np.nan
        pb = np.nan
        profit_margin = np.nan
        roe = np.nan
        debt_to_equity = np.nan

        try:
            stock = yf.Ticker(ticker)

            try:
                info = stock.get_info()
            except Exception:
                info = stock.info

            company = info.get("shortName", np.nan)
            sector = info.get("sector", "Unknown")
            market_cap = clean_number(info.get("marketCap", np.nan))
            pe = clean_number(info.get("trailingPE", np.nan))
            forward_pe = clean_number(info.get("forwardPE", np.nan))
            ps = clean_number(info.get("priceToSalesTrailing12Months", np.nan))
            pb = clean_number(info.get("priceToBook", np.nan))
            profit_margin = clean_number(info.get("profitMargins", np.nan))
            roe = clean_number(info.get("returnOnEquity", np.nan))
            debt_to_equity = clean_number(info.get("debtToEquity", np.nan))

        except Exception:
            pass

        rows.append({
            "Ticker": ticker,
            "Company": company,
            "Sector": sector,
            "Market Cap": market_cap,
            "P/E": pe,
            "Forward P/E": forward_pe,
            "P/S": ps,
            "P/B": pb,
            "Profit Margin": profit_margin,
            "ROE": roe,
            "Debt to Equity": debt_to_equity
        })

    return pd.DataFrame(rows)


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

    for col in ["P/E", "Forward P/E", "P/S", "P/B", "Profit Margin", "ROE", "Debt to Equity", "One Year Return"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    scored.loc[scored["P/E"] <= 0, "P/E"] = np.nan
    scored.loc[scored["Forward P/E"] <= 0, "Forward P/E"] = np.nan
    scored.loc[scored["P/S"] <= 0, "P/S"] = np.nan
    scored.loc[scored["P/B"] <= 0, "P/B"] = np.nan

    scored["P/E Score"] = percentile_score(scored["P/E"], higher_is_better=False)
    scored["Forward P/E Score"] = percentile_score(scored["Forward P/E"], higher_is_better=False)
    scored["P/S Score"] = percentile_score(scored["P/S"], higher_is_better=False)
    scored["P/B Score"] = percentile_score(scored["P/B"], higher_is_better=False)

    scored["Value Score"] = scored[
        ["P/E Score", "Forward P/E Score", "P/S Score", "P/B Score"]
    ].mean(axis=1)

    scored["Profit Margin Score"] = percentile_score(scored["Profit Margin"], higher_is_better=True)
    scored["ROE Score"] = percentile_score(scored["ROE"], higher_is_better=True)
    scored["Debt Score"] = percentile_score(scored["Debt to Equity"], higher_is_better=False)

    scored["Quality Score"] = scored[
        ["Profit Margin Score", "ROE Score", "Debt Score"]
    ].mean(axis=1)

    scored["Momentum Score"] = percentile_score(scored["One Year Return"], higher_is_better=True)

    scored["Final Score"] = (
        value_weight * scored["Value Score"] +
        quality_weight * scored["Quality Score"] +
        momentum_weight * scored["Momentum Score"]
    )

    scored["Final Score"] = scored["Final Score"].fillna(0)

    return scored.sort_values("Final Score", ascending=False)


def format_large_number(x):
    if pd.isna(x):
        return np.nan
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
        with st.spinner("Pulling stock prices and financial data..."):
            price_df = get_price_metrics(tickers)
            fundamental_df = get_fundamental_metrics(tickers)

            df = pd.merge(fundamental_df, price_df, on="Ticker", how="left")
            scored_df = calculate_scores(df)

        st.success("Stock screener completed.")

        st.subheader("Ranked Stock Screener Results")

        display_df = scored_df.copy()

        percent_cols = [
            "Profit Margin",
            "ROE",
            "One Year Return",
            "Annualized Volatility",
            "Value Score",
            "Quality Score",
            "Momentum Score",
            "Final Score"
        ]

        for col in percent_cols:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

        display_cols = [
            "Ticker",
            "Company",
            "Sector",
            "Market Cap",
            "Price",
            "P/E",
            "Forward P/E",
            "P/S",
            "P/B",
            "Profit Margin",
            "ROE",
            "Debt to Equity",
            "One Year Return",
            "Annualized Volatility",
            "Value Score",
            "Quality Score",
            "Momentum Score",
            "Final Score"
        ]

        st.dataframe(
            display_df[display_cols],
            use_container_width=True
        )

        missing_count = scored_df["Company"].isna().sum()

        if missing_count > 0:
            st.warning(
                f"{missing_count} ticker(s) returned limited company data from Yahoo Finance. "
                "The app still uses available price and metric data where possible."
            )

        top_stock = scored_df.iloc[0]

        st.subheader("Top Ranked Stock")

        st.write(
            f"**{top_stock['Ticker']}** ranks highest based on the selected scoring weights."
        )

        st.write(
            f"Final Score: **{top_stock['Final Score']:.2f}** | "
            f"Value Score: **{top_stock['Value Score']:.2f}** | "
            f"Quality Score: **{top_stock['Quality Score']:.2f}** | "
            f"Momentum Score: **{top_stock['Momentum Score']:.2f}**"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Top Stock", top_stock["Ticker"])

        with col2:
            st.metric("P/E Ratio", "N/A" if pd.isna(top_stock["P/E"]) else f"{top_stock['P/E']:.2f}")

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

        st.subheader("Value Score vs Quality Score")

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
            y="Quality Score",
            size="Market Cap Size",
            color="Sector",
            hover_name="Ticker",
            title="Value Score vs Quality Score"
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

        st.subheader("Valuation Multiples")

        valuation_df = scored_df[["Ticker", "P/E", "Forward P/E", "P/S", "P/B"]].copy()
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
            "The model rewards stocks with cheaper valuation ratios, stronger profitability/ROE, lower debt, "
            "and better recent momentum."
        )

        st.write(
            "This does not mean the top-ranked stock is guaranteed to be a good investment. "
            "The screener is best used as a first-pass filtering tool before deeper financial analysis."
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
