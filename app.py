import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
st.title("SNBT RKA ITS – 2025 Subtest Explorer")

# Initialize user scores with defaults for the KDE display
user_scores = {}
for s in subtests:
    user_scores[s] = float(np.median(df_2025[s]))
user_avg = float(np.mean(list(user_scores.values())))

# ---------- Helpers ----------
def band_label(p25, p50, p75, s):
    if s >= p75: return "🟩 Above 75% (strong)"
    if s >= p50: return "🟨 Above median (competitive)"
    if s >= p25: return "🟧 Above 25% (possible)"
    return "🟥 Below 25% (stretch)"

def kde_fig(series, title, user_x=None):
    """Return a Plotly figure with a true KDE curve using scipy."""
    x = np.asarray(series.dropna(), dtype=float)
    # guard for degenerate case
    if x.std() == 0:
        xs = np.linspace(x.min()-1, x.max()+1, 200)
        ys = np.zeros_like(xs)
    else:
        kde = gaussian_kde(x)  # Scott's rule bandwidth
        padding = (x.max() - x.min()) * 0.1 or 1.0
        xs = np.linspace(x.min() - padding, x.max() + padding, 400)
        ys = kde(xs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="KDE"))
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]],
                             y=np.r_[ys, np.zeros_like(ys)],
                             fill="toself", opacity=0.2, line=dict(width=0),
                             hoverinfo="skip", name=""))
    if user_x is not None:
        fig.add_vline(x=user_x, line_dash="dash",
                      annotation_text=f"Your score: {user_x:.1f}",
                      annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Score", yaxis_title="Density", showlegend=False)
    return fig

# ---------- KDE DISTRIBUTION ----------
st.header("Explore KDE Distribution")

# segmented control available in Streamlit 1.36+; fallback to radio if needed
try:
    selected = st.segmented_control("Choose subtest or avg",
                                    options=["avg"] + subtests,
                                    default="avg")
except Exception:
    selected = st.radio("Choose subtest or avg", options=["avg"] + subtests, index=0)

# Determine series & user value
if selected == "avg":
    series = df_2025["avg"]
    user_x = user_avg
else:
    series = df_2025[selected]
    user_x = user_scores[selected]

# Render KDE chart
fig = kde_fig(series, f"KDE Distribution of {selected} (2025)", user_x=user_x)
st.plotly_chart(fig, use_container_width=True)

# ---------- Input scores below KDE ----------
st.header("Enter Your Practice Scores")
cols = st.columns(len(subtests))
for i, s in enumerate(subtests):
    with cols[i]:
        default_val = user_scores[s]
        user_scores[s] = st.number_input(
            s, min_value=0.0, max_value=1000.0, value=default_val, step=1.0, key=f"input_{s}"
        )

user_avg = float(np.mean(list(user_scores.values())))
st.markdown(f"**Your avg (auto):** `{user_avg:.2f}`")

st.divider()

# ---------- KPI row ----------
st.subheader("Summary Statistics (2025)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean avg", f"{df_2025['avg'].mean():.2f}")
k2.metric("Median avg", f"{df_2025['avg'].median():.2f}")
k3.metric("Top quartile (p75)", f"{df_2025['avg'].quantile(0.75):.2f}")
k4.metric("Min–Max avg", f"{df_2025['avg'].min():.2f} – {df_2025['avg'].max():.2f}")

st.divider()

# ---------- Combined percentile table (subtests + avg highlighted) ----------
st.subheader("Percentiles & Your Band (2025 Data)")

# Build subtest percentiles
long = df_2025[subtests].melt(var_name="Subtest", value_name="Score")
sub_pct = (
    long.groupby("Subtest")["Score"]
    .quantile([0.25, 0.5, 0.75])
    .unstack()
    .rename(columns={0.25: "p25", 0.5: "p50", 0.75: "p75"})
    .round(2)
    .loc[subtests]
)
sub_pct["Your score"] = [round(user_scores[s], 2) for s in subtests]
sub_pct["Your band"] = [
    band_label(sub_pct.loc[s, "p25"], sub_pct.loc[s, "p50"], sub_pct.loc[s, "p75"], user_scores[s])
    for s in subtests
]

# Append avg row
p25, p50, p75 = (df_2025["avg"].quantile(q) for q in (0.25, 0.5, 0.75))
avg_band = band_label(p25, p50, p75, user_avg)
avg_row = pd.DataFrame(
    {"p25": [round(p25, 2)], "p50": [round(p50, 2)], "p75": [round(p75, 2)],
     "Your score": [round(user_avg, 2)], "Your band": [avg_band]},
    index=["avg"]
)
combined = pd.concat([sub_pct, avg_row], axis=0)

# Plotly table with highlighted avg row
header_vals = ["Subtest", "p25", "p50", "p75", "Your score", "Your band"]
index_col = combined.index.tolist()
cells_vals = [
    index_col,
    combined["p25"].astype(str).tolist(),
    combined["p50"].astype(str).tolist(),
    combined["p75"].astype(str).tolist(),
    combined["Your score"].astype(str).tolist(),
    combined["Your band"].tolist(),
]
row_colors = ["rgba(0,0,0,0)"] * (len(index_col) - 1) + ["rgba(255, 196, 0, 0.15)"]
fill_colors = [row_colors] * len(header_vals)
table = go.Figure(data=[go.Table(
    header=dict(values=header_vals, fill_color="rgba(80,80,80,0.6)", align="left"),
    cells=dict(values=cells_vals, fill_color=fill_colors, align="left")
)])
table.update_layout(title="Subtests + avg (highlighted)")
st.plotly_chart(table, use_container_width=True)

st.caption("This dashboard summarizes 2025 SNBT score data. Use as guidance, not a guarantee.")
