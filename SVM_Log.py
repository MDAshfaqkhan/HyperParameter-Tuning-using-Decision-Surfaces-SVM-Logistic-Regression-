import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions
from PIL import Image

# Load image for visualization
Hyp = Image.open("Hyperparameter.jpg")
st.image(Hyp, use_column_width=False)

# Title and file uploader
st.title("Dynamic Decision Surfaces: Hyperparameter Tuning with Logistic Regression & SVM")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    X = df.iloc[:, :-1].values  # All columns except the last for features
    y = df.iloc[:, -1].values   # The last column for target

    # Sidebar for algorithm selection
    algorithm = st.sidebar.selectbox("Select Algorithm", ("Logistic Regression", "SVM"))

    if algorithm == "Logistic Regression":
        # Hyperparameters for Logistic Regression
        C_value = st.sidebar.slider("Inverse of regularization strength (C)", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))
        penalty = st.sidebar.selectbox("Penalty", ("l1", "l2", "elasticnet", "none"))
        tol = st.sidebar.slider("Tolerance for stopping criteria (tol)", 0.001, 0.01, 0.001)
        max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200)
        random_state = st.sidebar.slider("Random State", 23, 42)
        l1_ratio = st.sidebar.slider("L1 Ratio (only for elasticnet)", 0.0, 1.0, 0.5)
        multiclass = st.sidebar.selectbox("Multiclass", ("auto", "ovr", "multinomial"))
        class_weight = st.sidebar.selectbox("Class Weight", (None, "balanced"))
        
        # Logistic Regression model
        model = LogisticRegression(
            C=C_value, 
            solver=solver, 
            penalty=penalty if solver != 'newton-cg' else 'l2',  
            tol=tol, 
            max_iter=max_iter,
            random_state=random_state, 
            l1_ratio=l1_ratio if penalty == 'elasticnet' else None, 
            multi_class=multiclass,
            class_weight=class_weight
        )
    elif algorithm == "SVM":
        # Hyperparameters for SVM
        C_value = st.sidebar.slider("C value", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))
        degree = st.sidebar.slider("Degree (for poly kernel)", 1, 5, 3)
        random_state = st.sidebar.slider("Random State", 23, 42)

        # SVM model
        model = SVC(C=C_value, kernel=kernel, degree=degree, random_state=random_state)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Display hyperparameters and metrics
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Hyperparameter Values")
        if algorithm == "Logistic Regression":
            st.write(f"C (Regularization strength): {C_value}")
            st.write(f"Solver: {solver}")
            st.write(f"Penalty: {penalty}")
            st.write(f"Tolerance (tol): {tol}")
            st.write(f"Max Iterations: {max_iter}")
            st.write(f"Random State: {random_state}")
            st.write(f"L1 Ratio: {l1_ratio}")
            st.write(f"Multiclass Strategy: {multiclass}")
            st.write(f"Class Weight: {class_weight}")
        else:
            st.write(f"C: {C_value}")
            st.write(f"Kernel: {kernel}")
            st.write(f"Degree (for poly kernel): {degree}")

    with col2:
        st.write("### Classification Metrics")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

    # Plot decision surface
    st.write("### Decision Surface")
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_train, y_train, clf=model, legend=2)
    plt.title(f'Decision Surface for {algorithm}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot(plt)
