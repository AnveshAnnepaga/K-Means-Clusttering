import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL STYLING (PROFESSIONAL DARK THEME)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0b0f19;
    color: #e5e7eb;
}

.block-container {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #f9fafb;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 12px;
}

.info-box {
    background-color: #0f172a;
    padding: 15px;
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    color: #d1d5db;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE & DESCRIPTION
# -------------------------------------------------
st.markdown("<h1>🟢 Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#9ca3af;'>This system uses <b>K-Means Clustering</b> "
    "to group customers based on their purchasing behavior and similarities.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("data/Wholesale customers data.csv")

numeric_features = df.select_dtypes(include="number").columns.tolist()

# -------------------------------------------------
# SIDEBAR - CLUSTERING CONTROLS
# -------------------------------------------------
st.sidebar.header("⚙️ Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    numeric_features,
    index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    numeric_features,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=5
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    value=42,
    step=1
)

run_clustering = st.sidebar.button("🟦 Run Clustering")

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
if run_clustering:

    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------
    st.markdown("<div class='section-title'>📊 Customer Clusters</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    cluster_colors = [
        "#22c55e", "#2563eb", "#f59e0b", "#ef4444",
        "#a855f7", "#06b6d4", "#ec4899", "#84cc16",
        "#eab308", "#14b8a6"
    ]

    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        ax.scatter(
            subset[feature_1],
            subset[feature_2],
            s=55,
            alpha=0.8,
            color=cluster_colors[c],
            label=f"Cluster {c}"
        )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="#000000",
        s=260,
        marker="X",
        edgecolors="white",
        linewidths=2,
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.legend()
    ax.set_facecolor("#ffffff")
    ax.grid(alpha=0.25)

    st.pyplot(fig)

    st.markdown("---")

    # -------------------------------------------------
    # CLUSTER SUMMARY
    # -------------------------------------------------
    st.markdown("<div class='section-title'>📋 Cluster Summary</div>", unsafe_allow_html=True)

    summary = df.groupby("Cluster").agg(
        Customers=("Cluster", "count"),
        Avg_Feature_1=(feature_1, "mean"),
        Avg_Feature_2=(feature_2, "mean")
    ).round(2)

    st.dataframe(summary)

    # -------------------------------------------------
    # BUSINESS INTERPRETATION
    # -------------------------------------------------
    st.markdown("<div class='section-title'>💡 Business Interpretation</div>", unsafe_allow_html=True)

    for c in summary.index:
        avg1 = summary.loc[c, "Avg_Feature_1"]
        avg2 = summary.loc[c, "Avg_Feature_2"]

        if avg1 > summary["Avg_Feature_1"].mean() and avg2 > summary["Avg_Feature_2"].mean():
            msg = "High-spending customers across selected categories"
            icon = "🟢"
        elif avg1 < summary["Avg_Feature_1"].mean() and avg2 < summary["Avg_Feature_2"].mean():
            msg = "Budget-conscious customers with lower spending"
            icon = "🟡"
        else:
            msg = "Moderate spenders with selective purchasing behavior"
            icon = "🔵"

        st.markdown(f"**{icon} Cluster {c}:** {msg}")

    # -------------------------------------------------
    # USER GUIDANCE
    # -------------------------------------------------
    st.markdown(
        "<div class='info-box'>"
        "📌 Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
        "</div>",
        unsafe_allow_html=True
    )

else:
    st.info("⬅️ Select features, choose K, and click **Run Clustering** to generate results.")
