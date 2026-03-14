import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 20px;
    }
    
    h1 {
        color: white !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-top: 1rem;
    }
    
    .sidebar .sidebar-content h2, 
    .sidebar .sidebar-content h3,
    .css-1d391kg h2,
    .css-1d391kg h3 {
        color: white !important;
    }
    
    .css-1d391kg, .css-1d391kg p, .css-1d391kg li {
        color: #e2e8f0 !important;
    }
    
    div[data-testid="stMetric"] {
        background: #1e293b !important;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }
    
    .css-1d391kg .stMarkdown p {
        color: #e2e8f0;
    }
    
    .css-1d391kg strong {
        color: #3b82f6;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
        width: 100%;
        font-size: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
    }
    
    .stSlider label {
        font-weight: 500 !important;
        color: white !important;
        font-size: 0.95rem !important;
    }
    
    .prediction-placeholder {
        background: transparent;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px dashed #3b82f6;
    }
    
    .prediction-placeholder h3 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .prediction-placeholder p {
        color: #94a3b8 !important;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 24px -8px rgba(37, 99, 235, 0.3);
    }
    
    .prediction-card h2 {
        color: white !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem;
        border-bottom: none;
    }
    
    .prediction-card p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: transparent;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #334155;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Iris Flower Classifier")
st.markdown("""
<div style='text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 1.1rem;'>
    Comparing multiple machine learning algorithms for iris species classification
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test, feature_names, target_names

X, y, X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()

@st.cache_resource
def train_and_compare_models(X_train, y_train, X_test, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
    
    return results, trained_models

model_results, trained_models = train_and_compare_models(X_train, y_train, X_test, y_test)

default_model = trained_models['Random Forest']
default_accuracy = model_results['Random Forest']['test_accuracy']

with st.sidebar:
    st.markdown("## Model Performance Comparison")
    
    comparison_data = []
    for name, metrics in model_results.items():
        comparison_data.append({
            'Model': name,
            'Test Accuracy': f"{metrics['test_accuracy']:.2%}",
            'CV Accuracy': f"{metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}"
        })
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("### Model Configuration")
    st.markdown(f"""
    **Selected Model:** Random Forest
    - **Trees:** 100
    - **Max Depth:** 5
    - **Training Samples:** {len(X_train)}
    - **Test Samples:** {len(X_test)}
    
    **Why Random Forest?**
    - Best accuracy among all models tested
    - Handles non-linear relationships well
    - Provides feature importance scores
    - Less prone to overfitting
    """)
    
    st.markdown("---")
    
    st.markdown("### Dataset Information")
    st.markdown("""
    **Iris Dataset** - 150 samples, 4 features:
    - Sepal Length (cm)
    - Sepal Width (cm)
    - Petal Length (cm)
    - Petal Width (cm)
    
    **Species:**
    - Setosa (0)
    - Versicolor (1)
    - Virginica (2)
    """)

tab1, tab2, tab3 = st.tabs(["Predictor", "Model Analysis", "Data Explorer"])

with tab1:
    st.header("Predict Iris Species")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Adjust Measurements")
        
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(X[:, 0].min()),
            max_value=float(X[:, 0].max()),
            value=float(X[:, 0].mean()),
            step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(X[:, 1].min()),
            max_value=float(X[:, 1].max()),
            value=float(X[:, 1].mean()),
            step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(X[:, 2].min()),
            max_value=float(X[:, 2].max()),
            value=float(X[:, 2].mean()),
            step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(X[:, 3].min()),
            max_value=float(X[:, 3].max()),
            value=float(X[:, 3].mean()),
            step=0.1,
            help="Width of the petal in centimeters"
        )
        
        selected_model_name = st.selectbox(
            "Select Model for Prediction",
            options=list(trained_models.keys()),
            index=0
        )
        
        predict_button = st.button("Predict Species", use_container_width=True)
    
    with col2:
        st.markdown("### Prediction Result")
        
        if predict_button:
            selected_model = trained_models[selected_model_name]
            
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            prediction = selected_model.predict(input_data)[0]
            if hasattr(selected_model, 'predict_proba'):
                prediction_proba = selected_model.predict_proba(input_data)[0]
            else:
                prediction_proba = [0, 0, 0]
                prediction_proba[prediction] = 1
            
            predicted_species = target_names[prediction]
            confidence = prediction_proba[prediction]
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>{predicted_species.title()}</h2>
                <p>Confidence: {confidence:.1%}</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">Model: {selected_model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Species': [name.title() for name in target_names],
                'Probability': prediction_proba
            })
            
            fig = px.bar(
                prob_df,
                x='Species',
                y='Probability',
                color='Species',
                color_discrete_sequence=['#3b82f6', '#2563eb', '#1d4ed8'],
                title="Class Probabilities"
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_font_color='white',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Input: SL={sepal_length}cm, SW={sepal_width}cm, PL={petal_length}cm, PW={petal_width}cm")
        else:
            st.markdown("""
            <div class="prediction-placeholder">
                <h3>👈 Adjust sliders and click Predict</h3>
                <p>The selected model will show the predicted species here</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("Model Performance Analysis")
    
    analysis_model = st.selectbox(
        "Select Model for Detailed Analysis",
        options=list(trained_models.keys()),
        index=0,
        key="analysis_model"
    )
    
    model = trained_models[analysis_model]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        
        fig_cm = px.imshow(
            cm,
            x=[name.title() for name in target_names],
            y=[name.title() for name in target_names],
            text_auto=True,
            color_continuous_scale='Blues',
            title=f'{analysis_model} - Accuracy: {accuracy:.2%}'
        )
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            title_font_size=14,
            font_color='white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Feature Analysis")
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': [name.replace(' (cm)', '') for name in feature_names],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                title="Feature Importance"
            )
            fig_imp.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="",
                title_font_size=14,
                font_color='white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"{analysis_model} doesn't provide feature importance scores.")
    
    st.subheader("Classification Report")
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    
    st.dataframe(report_df.style.format({
        'precision': '{:.2%}',
        'recall': '{:.2%}',
        'f1-score': '{:.2%}',
        'support': '{:.0f}'
    }), use_container_width=True)

with tab3:
    st.header("Explore the Iris Dataset")
    
    df = pd.DataFrame(X, columns=[name.replace(' (cm)', '') for name in feature_names])
    df['Species'] = [target_names[i].title() for i in y]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Species", len(target_names))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Pair Plot", "Box Plot", "Distribution Plot", "Correlation Heatmap"]
    )
    
    if viz_type == "Pair Plot":
        st.subheader("Pair Plot - Feature Relationships")
        fig = sns.pairplot(df, hue='Species', palette=['#3b82f6', '#2563eb', '#1d4ed8'])
        st.pyplot(fig)
        
    elif viz_type == "Box Plot":
        st.subheader("Box Plot - Feature Distribution by Species")
        feature = st.selectbox("Select Feature", [col for col in df.columns if col != 'Species'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Species', y=feature, data=df, palette=['#3b82f6', '#2563eb', '#1d4ed8'])
        plt.title(f'Distribution of {feature} by Species')
        plt.xlabel('Species')
        plt.ylabel(feature)
        st.pyplot(fig)
        
    elif viz_type == "Distribution Plot":
        st.subheader("Distribution Plot - Feature Distribution")
        feature = st.selectbox("Select Feature", [col for col in df.columns if col != 'Species'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for species in df['Species'].unique():
            subset = df[df['Species'] == species]
            sns.kdeplot(data=subset[feature], label=species)
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.title(f'Distribution of {feature} by Species')
        plt.legend()
        st.pyplot(fig)
        
    else:
        st.subheader("Correlation Heatmap")
        
        corr_df = df.drop('Species', axis=1)
        corr_matrix = corr_df.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            title="Feature Correlations"
        )
        fig_corr.update_layout(
            title_font_size=14,
            font_color='white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    Built with Streamlit • Multiple Model Comparison • Iris Flower Classification • 2026
</div>
""", unsafe_allow_html=True)