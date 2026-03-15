import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("Country Clustering Dashboard")
st.markdown("""
This interactive dashboard clusters countries based on economic and development indicators.

*Techniques Used*
- K-Means Clustering
- PCA (Principal Component Analysis)
- Clustering Evaluation Metrics

The goal is to identify patterns and similarities between countries using data-driven clustering.
""")
# Load dataset
data = pd.read_csv("WDM.csv")
st.subheader("Project Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Countries", data["Country"].nunique())
col2.metric("Column Used", len(data.columns))
col3.metric("Dataset Rows", len(data))
import numpy as np
data = data.replace("None", np.nan)
st.subheader("Dataset Preview")
st.download_button(
    label="Download Dataset",
    data=data.to_csv(index=False),
    file_name="country_dataset.csv",
    mime="text/csv"
)
st.dataframe(data.head())

# Load model and scaler
model = pickle.load(open("kmeans_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Features used in clustering
features = [
"Birth Rate",
"CO2 Emissions",
"Days to Start Business",
"Ease of Business",
"Energy Usage",
"GDP",
"Health Exp % GDP",
"Health Exp/Capita",
"Hours to do Tax",
"Infant Mortality Rate",
"Internet Usage",
"Lending Interest",
"Life Expectancy Female",
"Life Expectancy Male",
"Mobile Phone Usage",
"Population 0-14",
"Population 15-64",
"Population 65+",
"Population Urban"
]
numeric_data = data[features]
# remove $ and commas then convert to numeric
for col in numeric_data.columns:
    numeric_data[col] = numeric_data[col].astype(str)
    numeric_data[col] = numeric_data[col].str.replace('$','',regex=False)
    numeric_data[col] = numeric_data[col].str.replace(',','',regex=False)
    numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')

# handle missing values
numeric_data = numeric_data.fillna(numeric_data.median())

# Scale data
scaled = scaler.transform(numeric_data)

# Predict clusters
clusters = model.predict(scaled)
data["Cluster"] = clusters
st.subheader("Cluster Distribution")
cluster_counts = data["Cluster"].value_counts().sort_index()
st.write(cluster_counts)
import matplotlib.pyplot as plt
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values)
ax_bar.set_xlabel("Cluster")
ax_bar.set_ylabel("Number of Countries")
ax_bar.set_title("Cluster Distribution")
st.pyplot(fig_bar)

# Clustering Evaluation Scores
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
st.subheader("Clustering Evaluation Scores")
sil_score = silhouette_score(scaled, clusters)
db_score = davies_bouldin_score(scaled, clusters)
ch_score = calinski_harabasz_score(scaled, clusters)
col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score", round(sil_score,3))
col2.metric("Davies-Bouldin Score", round(db_score,3))
col3.metric("Calinski-Harabasz Score", round(ch_score,3))
st.subheader("Cluster Statistics")
cluster_stats = data.groupby("Cluster").mean(numeric_only=True)
st.write(cluster_stats)
st.sidebar.header("Cluster Explorer")
selected_cluster = st.sidebar.selectbox(
    "Select Cluster",
    sorted(data["Cluster"].unique()),
    key="Cluster_select_sidebar"
)
st.sidebar.header("Country Search")
country_name = st.sidebar.text_input("Enter Country Name")
if country_name:
    result = data[data["Country"].str.contains(country_name, case=False, na=False)]
    st.subheader("Country Details")
    st.write(result)
cluster_countries = data[data["Cluster"] == selected_cluster]
st.subheader(f"Countries in Cluster {selected_cluster}")
st.write(cluster_countries[["Country"]])

# PCA visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)
pca_df = pd.DataFrame(pca_data, columns=["PC1","PC2"])
pca_df["Cluster"] = clusters
fig, ax = plt.subplots()
for c in pca_df["Cluster"].unique():
    subset = pca_df[pca_df["Cluster"] == c]
    ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {c}")
ax.set_title("Country Clusters (PCA)")
ax.legend()
st.pyplot(fig)
st.subheader("Countries by Cluster")
cluster_selected = st.selectbox(
    "Select Cluster",
    sorted(data["Cluster"].unique())
)
cluster_countries = data[data["Cluster"] == cluster_selected]["Country"]
st.write(cluster_countries)

st.subheader("Model Information")

st.write("""
Model Used: *K-Means Clustering*

Evaluation Metrics Used:
- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score

Dimensionality Reduction:
- PCA (2 Components) for visualization

Deployment:
- Built with Streamlit
- Hosted on Streamlit Cloud
""")