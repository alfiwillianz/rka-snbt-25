# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# ---------- Inputs at TOP ----------
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
st.markdown(f"**Your avg (auto):** `{user_avg:.2f}`")

st.divider()

# ---------- Helpers ----------
def band_label(p25, p50, p75, s):
    if s >= p75: return "🟩 Above 75% (strong)"
    if s >= p50: return "🟨 Above median (competitive)"
    if s >= p25: return "🟧 Above 25% (possible)"
    return "🟥 Below 25% (stretch)"

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean avg (2025)", f"{df_2025['avg'].mean():.2f}")
k2.metric("Median avg (p50)", f"{df_2025['avg'].median():.2f}")
k3.metric("Top quartile (p75)", f"{df_2025['avg'].quantile(0.75):.2f}")
k4.metric("Min–Max avg", f"{df_2025['avg'].min():.2f} – {df_2025['avg'].max():.2f}")

# ---------- "KDE-like" Distribution of avg (no SciPy needed) ----------
fig_kde = px.histogram(
    df_2025, x="avg", nbins=30, histnorm="probability density",
    opacity=0.75, marginal="violin", title="Distribution of avg (2025, density)"
)
fig_kde.add_vline(x=user_avg, line_dash="dash", annotation_text=f"Your avg: {user_avg:.1f}")
st.plotly_chart(fig_kde, use_container_width=True)

# ---------- ECDF of avg ----------
rr_sorted = np.sort(df_2025["avg"].values)
user_percentile = (rr_sorted <= user_avg).sum() / len(rr_sorted)
fig_ecdf = px.ecdf(df_2025, x="avg", title="ECDF of avg (Percentiles, 2025)")
fig_ecdf.add_vline(x=user_avg, line_dash="dash")
fig_ecdf.add_annotation(x=user_avg, y=user_percentile,
                        text=f"Your percentile ≈ {user_percentile*100:.1f}%",
                        showarrow=True)
st.plotly_chart(fig_ecdf, use_container_width=True)

# ---------- Boxplot per subtest ----------
long_2025 = df_2025[subtests].melt(var_name="Subtest", value_name="Score")
fig_box = px.box(long_2025, x="Subtest", y="Score", title="Subtest Score Spread (2025)")
st.plotly_chart(fig_box, use_container_width=True)

# ---------- Density per subtest (no SciPy) ----------
st.subheader("Subtest Distributions (2025, density)")
for s in subtests:
    fig_sub = px.histogram(
        df_2025, x=s, nbins=30, histnorm="probability density",
        opacity=0.75, marginal="violin", title=f"Distribution of {s} (2025, density)"
    )
    fig_sub.add_vline(x=user_scores[s], line_dash="dash",
                      annotation_text=f"Your {s}: {user_scores[s]:.1f}")
    st.plotly_chart(fig_sub, use_container_width=True)

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
sub_pct["Your score"] = [round(user_scores[s], 2) for s in sub_pct.index]
sub_pct["Your band"] = [
    band_label(sub_pct.loc[s, "p25"], sub_pct.loc[s, "p50"], sub_pct.loc[s, "p75"], user_scores[s])
    for s in sub_pct.index
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

# Plotly table so we can highlight avg row
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

row_colors = ["rgba(0,0,0,0)"] * (len(index_col) - 1) + ["rgba(255, 196, 0, 0.15)"]  # highlight avg
fill_colors = [row_colors] * len(header_vals)

table = go.Figure(
    data=[
        go.Table(
            header=dict(values=header_vals, fill_color="rgba(80,80,80,0.6)", align="left"),
            cells=dict(values=cells_vals, fill_color=fill_colors, align="left")
        )
    ]
)
table.update_layout(title="Subtests + avg (highlighted)")
st.plotly_chart(table, use_container_width=True)

st.caption("This dashboard summarizes 2025 SNBT subtest data. Use as guidance, not a guarantee.")
