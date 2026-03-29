import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄", layout="centered")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

@st.cache_data
def load_data():
    # Update this path if needed
    data = pd.read_csv(r'C:\Users\91995\OneDrive\Documents\Streamlit Project\mushrooms.csv')

    # Encode all categorical columns
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])

    return data


@st.cache_data
def split_data(df):
    # If target column is 'type', use it. Otherwise use first column
    target_col = 'type' if 'type' in df.columns else df.columns[0]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    return train_test_split(X, y, test_size=0.3, random_state=0)


def plot_metrics(metrics_list, model, X_test, y_test):
    class_names = ['edible', 'poisonous']

    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=class_names, ax=ax
        )
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)


def display_results(model, X_test, y_test, metrics_list):
    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.success("Model trained successfully!")

    st.write("### Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    plot_metrics(metrics_list, model, X_test, y_test)


# -------------------------------
# MAIN APP
# -------------------------------

def main():
    st.title("🍄 Mushroom Classification Web App")
    st.write("Predict whether a mushroom is **edible** or **poisonous** using Machine Learning.")

    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Sidebar
    st.sidebar.title("⚙️ Model Settings")

    classifier = st.sidebar.selectbox(
        "Choose Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
    )

    metrics = st.sidebar.multiselect(
        "Select Metrics to Plot",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
    )

    # -------------------------------
    # SVM
    # -------------------------------
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("SVM Hyperparameters")

        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))

        if st.sidebar.button("Train & Classify"):
            st.subheader("SVM Results")

            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(X_train, y_train)

            display_results(model, X_test, y_test, metrics)

    # -------------------------------
    # LOGISTIC REGRESSION
    # -------------------------------
    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Logistic Regression Hyperparameters")

        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0, key="lr_c")
        max_iter = st.sidebar.slider("Maximum Iterations", 100, 500, value=200)

        if st.sidebar.button("Train & Classify"):
            st.subheader("Logistic Regression Results")

            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)

            display_results(model, X_test, y_test, metrics)

    # -------------------------------
    # RANDOM FOREST
    # -------------------------------
    elif classifier == "Random Forest":
        st.sidebar.subheader("Random Forest Hyperparameters")

        n_estimators = st.sidebar.slider("Number of Trees", 100, 500, value=100)
        max_depth = st.sidebar.slider("Maximum Depth", 1, 20, value=10)

        if st.sidebar.button("Train & Classify"):
            st.subheader("Random Forest Results")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=0
            )
            model.fit(X_train, y_train)

            display_results(model, X_test, y_test, metrics)

    # -------------------------------
    # SHOW DATASET
    # -------------------------------
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Encoded Mushroom Dataset")
        st.dataframe(df)


if __name__ == "__main__":
    main()