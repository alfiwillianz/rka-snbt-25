import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.set_page_config(page_title="SNBT RKA ITS – 2025 Subtest Explorer", layout="wide")

# ---------- 2025 baseline data (hardcoded) ----------
data_2025 = {
    "PU":  [794.17,726.01,703.73,808.27,821.37,721.33,795.85,702.38,644.26,753.10,711.74,727.46,637.82,704.38,805.14,739.70,762.19],
    "PPU": [775.25,731.23,643.17,753.40,695.09,727.57,634.10,684.77,637.92,708.85,622.25,706.04,638.25,775.30,700.33,661.21,657.78],
    "PBM": [755.89,677.02,712.31,774.88,666.16,693.37,726.09,687.94,663.24,766.61,814.85,636.70,720.01,697.87,596.62,644.09,690.64],
    "PK":  [811.72,711.98,792.57,864.97,755.88,764.31,760.52,809.87,820.26,679.28,811.44,739.52,820.15,733.76,735.20,798.57,867.68],
    "LBI": [545.32,781.21,714.82,672.85,648.64,635.72,751.95,656.71,703.52,692.22,722.90,719.67,713.21,793.93,648.64,726.12,695.45],
    "LBE": [671.48,749.70,759.55,766.41,669.42,691.94,717.70,746.23,782.71,788.69,758.78,805.77,804.42,762.57,749.70,733.50,698.95],
    "PM":  [660.28,633.32,651.71,761.07,774.04,790.31,623.09,756.42,795.70,861.76,882.04,843.63,832.05,876.79,835.38,673.32,731.21],
}
subtests = ["PU","PPU","PBM","PK","LBI","LBE","PM"]
df_2025 = pd.DataFrame(data_2025)
df_2025["avg"] = df_2025[subtests].mean(axis=1)

# ---------- Title ----------
st.title("SNBT RKA ITS – 2025 Score")
st.caption("Based on 2025 SNBT score data. Use as guidance, not a guarantee.")

# ---------- Summary Statistics ----------
st.header("Summary Statistics (2025)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean avg", f"{df_2025['avg'].mean():.2f}")
k2.metric("Median avg", f"{df_2025['avg'].median():.2f}")
k3.metric("Top quartile (p75)", f"{df_2025['avg'].quantile(0.75):.2f}")
k4.metric("Min–Max avg", f"{df_2025['avg'].min():.2f} – {df_2025['avg'].max():.2f}")

# ---------- Detailed Statistics Table ----------
st.header("Detailed Statistics by Subtest")
stats_data = {}
all_columns = subtests + ["avg"]

for col in all_columns:
    stats_data[col] = [
        df_2025[col].min(),
        df_2025[col].quantile(0.25),
        df_2025[col].median(),
        df_2025[col].quantile(0.75),
        df_2025[col].max(),
        df_2025[col].mean()
    ]

stats_df = pd.DataFrame(stats_data, 
                       index=["Min", "25%", "Median", "75%", "Max", "Mean"])

# Display the table with full width
st.dataframe(stats_df, use_container_width=True)

st.divider()

# ---------- Practice Score Input ----------
st.header("Enter Your Practice Scores")
cols = st.columns(len(subtests))
user_scores = {}
for i, s in enumerate(subtests):
    with cols[i]:
        default_val = float(np.median(df_2025[s]))
        user_scores[s] = st.number_input(
            s, min_value=0.0, max_value=1000.0, value=default_val, step=1.0
        )

user_avg = float(np.mean(list(user_scores.values())))
st.markdown(f"**Your avg:** `{user_avg:.2f}`")

st.divider()

# ---------- Helper function ----------
def kde_fig(series, title, user_x=None):
    """Return a Plotly figure with a true KDE curve using scipy."""
    x = np.asarray(series.dropna(), dtype=float)
    
    # Calculate percentiles for the current series
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    
    # Determine user's performance band
    band = "No user score provided" # Default
    band_color = "grey"
    if user_x is not None:
        if user_x >= p75:
            band = "🟩 Above 75% (Strong)"
            band_color = "green"
        elif user_x >= p50:
            band = "🟨 Above Median (Competitive)"
            band_color = "orange"
        elif user_x >= p25:
            band = "🟧 Above 25% (Possible)"
            band_color = "darkorange"
        else:
            band = "🟥 Below 25% (Stretch)"
            band_color = "red"
    
    # guard for degenerate case
    if x.std() == 0:
        xs = np.linspace(x.min()-1, x.max()+1, 200)
        ys = np.zeros_like(xs)
    else:
        kde = gaussian_kde(x)  # Scott's rule bandwidth

        # Dynamically calculate plot range to include user's score
        plot_min = x.min()
        plot_max = x.max()
        if user_x is not None:
            plot_min = min(plot_min, user_x)
            plot_max = max(plot_max, user_x)

        padding = (plot_max - plot_min) * 0.1 or 1.0
        xs = np.linspace(plot_min - padding, plot_max + padding, 400)
        ys = kde(xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="KDE", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]],
                             y=np.r_[ys, np.zeros_like(ys)],
                             fill="toself", opacity=0.2, line=dict(width=0, color="blue"),
                             hoverinfo="skip", name="", showlegend=False))
    
    fig.add_vline(x=p25, line=dict(color="red", dash="dot", width=1), 
                  annotation_text="25%", annotation_position="top")
    fig.add_vline(x=p50, line=dict(color="orange", dash="dot", width=1),
                  annotation_text="50%", annotation_position="top")
    fig.add_vline(x=p75, line=dict(color="green", dash="dot", width=1),
                  annotation_text="75%", annotation_position="top")
    
    if user_x is not None:
        fig.add_vline(x=user_x, line=dict(color=band_color, dash="dash", width=3),
                      annotation_text=f"Your score: {user_x:.1f}",
                      annotation_position="top left")
        
        fig.update_layout(
            title=f"{title}<br><span style='font-size:14px; color:{band_color}'>{band}</span>",
            xaxis_title="Score", 
            yaxis_title="Density", 
            showlegend=False
        )
    else:
        fig.update_layout(title=title, xaxis_title="Score", yaxis_title="Density", showlegend=False)
    
    return fig

# ---------- Visualization Analysis ----------
st.header("KDE Distribution Analysis")


selected = st.segmented_control("Choose subtest or avg", options=["avg"] + subtests, default="avg")

# Determine series & user value
if selected == "avg":
    series = df_2025["avg"]
    user_x = user_avg
else:
    series = df_2025[selected]
    user_x = user_scores[selected]

# Create and display the KDE plot
kde_fig_plot = kde_fig(series, f"KDE Distribution of {selected} (2025)", user_x=user_x)
st.plotly_chart(kde_fig_plot, use_container_width=True)