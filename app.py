import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide"
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

MODEL_PATH = (
    "outputs/clustering_artifacts.joblib"
)


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


artifacts = load_model()

imputer = artifacts["imputer"]

scaler = artifacts["scaler"]

pca = artifacts["pca"]

model = artifacts["model"]

model_name = artifacts["model_name"]

model_parameters = (
    artifacts["model_parameters"]
)

feature_columns = (
    artifacts["feature_columns"]
)


# ---------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------

def clean_numeric_value(value):

    if pd.isna(value):

        return np.nan

    if isinstance(value, str):

        value = value.strip()

        value = value.replace(
            "$",
            ""
        )

        value = value.replace(
            ",",
            ""
        )

        value = value.replace(
            "%",
            ""
        )

    return pd.to_numeric(
        value,
        errors="coerce"
    )


# ---------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------

def prepare_data(
    raw_data
):

    if "Country" not in raw_data.columns:

        raise ValueError(
            "Dataset must contain Country column."
        )

    data = raw_data.copy()

    for column in data.columns:

        if column != "Country":

            data[column] = (
                data[column]
                .apply(
                    clean_numeric_value
                )
            )

    country_data = (
        data
        .groupby("Country")
        .median(
            numeric_only=True
        )
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in country_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            +
            ", ".join(
                missing_features
            )
        )

    country_data = (
        country_data[
            feature_columns
        ]
    )

    return country_data


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(
    "🌍 World Development Measurement"
)

st.header(
    "Country Clustering Application"
)

st.write(
    """
This application groups countries according to
economic, health, demographic and development
indicators.
"""
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    f"Model: **{model_name}**"
)

st.sidebar.write(
    f"Parameters: **{model_parameters}**"
)

st.sidebar.write(
    f"Features: **{len(feature_columns)}**"
)

st.sidebar.write(
    f"PCA Components: **{pca.n_components_}**"
)


# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload World Development Dataset",
    type=[
        "xlsx",
        "xls",
        "csv"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload an Excel or CSV file."
    )

    st.stop()


# ---------------------------------------------------
# READ FILE
# ---------------------------------------------------

if uploaded_file.name.endswith(
    ".csv"
):

    raw_data = pd.read_csv(
        uploaded_file
    )

else:

    raw_data = pd.read_excel(
        uploaded_file
    )


st.subheader(
    "Uploaded Dataset"
)

st.dataframe(
    raw_data.head(20),
    use_container_width=True
)


# ---------------------------------------------------
# PROCESS DATA
# ---------------------------------------------------

try:

    country_data = prepare_data(
        raw_data
    )

    X = country_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_imputed = pd.DataFrame(

        imputer.transform(X),

        columns=feature_columns,

        index=country_data.index
    )


    # Log transformation

    X_transformed = (
        X_imputed.copy()
    )

    for column in (
        X_transformed.columns
    ):

        if (
            X_transformed[column]
            >= 0
        ).all():

            X_transformed[column] = (
                np.log1p(
                    X_transformed[column]
                )
            )


    # Scaling

    X_scaled = (
        scaler.transform(
            X_transformed
        )
    )


    # PCA

    X_pca = (
        pca.transform(
            X_scaled
        )
    )


    # Prediction

    labels = (
        model.predict(
            X_pca
        )
    )


    result = (
        country_data.copy()
    )

    result[
        "Cluster"
    ] = labels


except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )

    st.stop()


# ---------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------

col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Countries",
    len(result)
)


col2.metric(
    "Clusters",
    result[
        "Cluster"
    ].nunique()
)


col3.metric(
    "Features",
    len(feature_columns)
)


col4.metric(
    "PCA Components",
    pca.n_components_
)


# ---------------------------------------------------
# CLUSTER DISTRIBUTION
# ---------------------------------------------------

st.subheader(
    "Cluster Distribution"
)

cluster_counts = (
    result[
        "Cluster"
    ]
    .value_counts()
    .sort_index()
)

st.bar_chart(
    cluster_counts
)


# ---------------------------------------------------
# COUNTRY EXPLORER
# ---------------------------------------------------

st.subheader(
    "Country Explorer"
)

selected_country = (
    st.selectbox(
        "Select Country",
        result.index.tolist()
    )
)

country_row = (
    result.loc[
        selected_country
    ]
)


st.metric(
    "Assigned Cluster",
    str(
        country_row[
            "Cluster"
        ]
    )
)


st.dataframe(
    country_row
    .drop("Cluster")
    .to_frame("Value"),
    use_container_width=True
)


# ---------------------------------------------------
# ALL RESULTS
# ---------------------------------------------------

st.subheader(
    "Country Cluster Assignments"
)

st.dataframe(
    result[
        ["Cluster"]
    ].sort_values(
        "Cluster"
    ),
    use_container_width=True
)


# ---------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------

csv_data = (
    result
    .reset_index()
    .to_csv(
        index=False
    )
)


st.download_button(
    "Download Cluster Results",
    csv_data,
    "country_cluster_results.csv",
    "text/csv"
)