# streamlit_app.py
"""
Streamlit app for:
"Wikipedia Engagement with Climate Change Topics in South Asia"

Author: Soha Khan
Created from notebook: 06_my_RQ.ipynb
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page config 
st.set_page_config(page_title="Wiki CC — South Asia", layout="wide")

# Title & RQ 
st.title("Wikipedia Engagement with Climate Change Topics in South Asia")
st.markdown(
    """
**Author:** Soha Khan  
**Final Research Question:**  
**How visible are climate-related Wikipedia topics across South Asia, and what does the lack of data for Pakistan suggest about public engagement with climate change online?**

**Goal:** Compare daily Wikipedia pageviews on climate change topics (2020–2025) across South Asian countries and investigate why some countries (e.g. Pakistan) may be missing from the dataset.
"""
)

# Data description 
st.header("Data used & preparation")
st.markdown(
    """
- **Source file:** 'data/daily_cc_interest_total_per_country.csv'  
  Each row in this CSV should have: 'date, country, total_cc_pageview_counts'.
- **Preparation steps performed by this app:**
  1. Load the CSV.
  2. Parse 'date' column to datetime.
  3. Filter to the selected date range and countries.
  4. Aggregate daily totals by country.
  5. Provide options for smoothing and log-transform for visualization.
"""
)

# Load data the same way as the working sample
if 'student_data' not in st.session_state or st.session_state['student_data']['st12_df'].empty:
    st.warning("Data not loaded. Please ensure the main Home Page ran successfully and the data files exist.")
else:
    df = st.session_state['student_data']['st12_df']

# Load data the same way as the working sample 

df = df.rename(columns={c: c.strip() for c in df.columns})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["total_cc_pageview_counts"] = (pd.to_numeric(df["total_cc_pageview_counts"], errors="coerce").fillna(0).astype(int))



# Sidebar: interactive controls 
st.sidebar.header("Filters & plot options")

# Determine default countries: use South Asia list, but choose only those present
south_asia_default = ["India", "Bangladesh", "Nepal", "Sri Lanka", "Pakistan"]
present_countries = sorted(df["country"].unique())
# ensure defaults are in present list
default_selected = [c for c in south_asia_default if c in present_countries]
if not default_selected:
    # fallback
    default_selected = present_countries[:3]

countries = st.sidebar.multiselect(
    "Select countries to show",
    options=present_countries,
    default=default_selected
)

# Date range picker
min_date = df["date"].min().date()
max_date = df["date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
# smoothing and log options
smooth_days = st.sidebar.slider("Smoothing window (days, 0 = none)", 0, 30, 7)
use_log = st.sidebar.checkbox("Plot log10(pageviews)", value=False)
normalize_per_100k = st.sidebar.checkbox("Normalize per 100k people (requires population file)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If a country is missing from the plot, it likely does not appear in the dataset (no recorded pageviews or excluded for privacy).")

# Data filtering & aggregation 
st.header("Time series visualization")

# Validate date inputs
if isinstance(start_date, tuple) or start_date is None:
    # if user selected single date accidentally, fix
    start_date = min_date
    end_date = max_date

# Convert to datetime
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

if start_dt > end_dt:
    st.error("Start date must be before end date.")
    st.stop()

# Filter by date & countries
mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
if countries:
    mask = mask & (df["country"].isin(countries))
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data available for the selected countries/date range.")
else:
    # aggregate to daily totals per country (defensive)
    agg = filtered.groupby(["date", "country"], as_index=False)["total_cc_pageview_counts"].sum()

    # optionally normalize per population (if user requested)
    if normalize_per_100k:
        pop_path = "data/country_region_population.csv"
        if os.path.exists(pop_path):
            pop_df = pd.read_csv(pop_path)
            pop_df["country"] = pop_df["country"].astype(str).str.strip()
            pop_df["population"] = pd.to_numeric(pop_df["population"], errors="coerce")
            agg = agg.merge(pop_df[["country", "population"]], on="country", how="left")
            if agg["population"].isnull().any():
                st.warning("Population missing for some selected countries; those will not be normalized.")
            agg["total_cc_pageview_counts_per100k"] = agg.apply(
                lambda r: (r["total_cc_pageview_counts"] / r["population"] * 100000) if pd.notnull(r.get("population")) and r["population"] > 0 else np.nan,
                axis=1
            )
            y_col = "total_cc_pageview_counts_per100k"
            y_label = "Pageviews per 100k people"
        else:
            st.warning("Population mapping file not found at data/country_region_population.csv. Uncheck normalization or add the file.")
            y_col = "total_cc_pageview_counts"
            y_label = "Total Pageviews"
    else:
        y_col = "total_cc_pageview_counts"
        y_label = "Total Pageviews"

    # Pivot for plotting convenience
    pivot = agg.pivot(index="date", columns="country", values=y_col).fillna(0).sort_index()

    # Apply smoothing if requested
    if smooth_days > 0:
        pivot = pivot.rolling(window=smooth_days, min_periods=1, center=True).mean()

    # Use log transform if requested
    plot_data = pivot.copy()
    if use_log:
        # add small epsilon to avoid log(0)
        plot_data = np.log10(plot_data.replace(0, np.nan))
        # replace -inf/inf back to NaN and then fill with very small number for plotting
        plot_data = plot_data.replace([-np.inf, np.inf], np.nan)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_data.plot(ax=ax, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label + (" (log10)" if use_log else ""))
    ax.set_title("Daily Wikipedia Pageviews on Climate Change Topics")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)

    # Quick interpretation block
    st.markdown("### Quick interpretation")
    st.markdown(
        """
        - The plot shows daily time series of climate-change pageviews for the selected countries and date range.
        - If a country is completely missing, it likely has no recorded pageviews in the dataset (this is common for some countries, including Pakistan in this dataset).
        - Use smoothing or log-scale to better compare countries with different magnitudes of activity.
        """
    )

# Data snippet & download 
st.header("Data snapshot & export")
st.markdown("Below is a small snippet of the filtered data used for the plot. You can download the full filtered CSV.")

if not filtered.empty:
    # show a few rows
    st.dataframe(filtered.sort_values(["date", "country"]).head(200))
    # allow download
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_daily_cc_pageviews.csv", mime="text/csv")
else:
    st.write("No rows to display for selected filters.")

# Final notes 
st.markdown("---")
st.markdown(
    """
    ## Notes & next steps
    - The absence of country data (for example Pakistan) in the DPDP dataset can reflect either:
      1. Very low Wikipedia usage in that country, or
      2. Privacy thresholds / data exclusion for small traffic numbers.
    - Possible extensions for the app:
      - Merge in population & region data to produce per-capita comparisons (a toggle is provided).
      - Overlay event windows (e.g., major floods, heatwaves) as shaded regions on the time series.
      - Add month / year aggregation views for clearer long-term trends.
    """
)
