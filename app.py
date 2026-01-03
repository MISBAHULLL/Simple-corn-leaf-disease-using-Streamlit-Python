import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from modules.pipeline import predict_image_with_model, get_class_names
from modules.model_loader import load_model, get_available_models
from modules.utils import CLASS_MAP, CLASS_COLORS, CLASS_DESCRIPTIONS
from modules.visualization import (
    generate_confusion_matrix,
    generate_model_comparison_chart,
    generate_model_comparison_table,
    generate_metrics_table,
    generate_distribution_plot,
    generate_feature_importance_plot,
    generate_shap_summary_plot,
    generate_shap_force_plot,
    generate_shap_dependence_plot,
    generate_evaluation_metrics_table,
    generate_data_class_table,
    generate_top_features_table
)
from modules.evaluation_data import DATASET_DISTRIBUTION, METRICS, ACCURACIES, TOTAL_IMAGES

# === PAGE CONFIG ===
st.set_page_config(
    page_title="CornShield - Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === GLOBAL STYLES (Professional & Clean) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #16a34a;
        --primary-dark: #15803d;
        --primary-light: #dcfce7;
        --accent: #eab308;
        --bg-main: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
        --radius: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg-main);
    }

    /* Hero Section - Compact */
    .hero {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: var(--radius);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: "🌽";
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 4rem;
        opacity: 0.15;
    }

    .hero h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }

    .hero p {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
        max-width: 500px;
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }

    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }

    .stat-item {
        background: var(--bg-main);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }

    .stat-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, var(--primary-light) 0%, #fff 100%);
        border: 2px solid var(--primary);
        border-radius: var(--radius);
        padding: 1.5rem;
        text-align: center;
    }

    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .result-class {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
    }

    .result-confidence {
        display: inline-block;
        background: var(--primary);
        color: white;
        padding: 0.25rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Image Container - Controlled Size */
    .img-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
        text-align: center;
    }

    .img-container img {
        max-width: 100%;
        border-radius: 8px;
    }

    .img-caption {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }

    /* Progress Bar */
    .prob-bar {
        margin-bottom: 0.75rem;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .prob-track {
        background: var(--border);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
    }

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        background: var(--primary-light);
        border: 2px dashed var(--primary);
        border-radius: var(--radius);
    }
</style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "XGBoost (Best)"

# === HELPER FUNCTIONS ===
def get_cropped_vis_path():
    """Get the cropped visualisasi folder path."""
    return os.path.join(os.path.dirname(__file__), "visualisasi", "cropped")

def get_vis_path():
    """Get the visualisasi folder path."""
    return os.path.join(os.path.dirname(__file__), "visualisasi")

def get_sample_images_path():
    """Get the sample images folder path."""
    return os.path.join(os.path.dirname(__file__), "assets", "sample_images")

def display_image_card(img_path, caption="", max_width=600):
    """Display image in a nice card container with controlled width."""
    if os.path.exists(img_path):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(img_path, caption=caption, width=max_width)


# === PAGE: HOME ===
def render_home_page():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🌽 Corn Leaf Disease Classifier</h1>
        <p>AI-powered diagnosis for corn leaf diseases. Detects Healthy, Damaged, Blight, and Rust conditions with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-value">4</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">313</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">ML Models</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">94%+</div>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns: Project info & Pipeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📋 About This Project</div>
            <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                <strong>CornShield</strong> is a machine learning system for automatic corn leaf disease classification.
                It uses texture feature extraction (LBP, Gradient, DOR) combined with ensemble classifiers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <div class="card-title">🧬 Processing Pipeline</div>
            <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                1. <strong>Preprocessing</strong> → Resize 256×256, Normalize<br>
                2. <strong>Segmentation</strong> → Otsu Thresholding<br>
                3. <strong>Features</strong> → Fine LBP + Coarse + DOR<br>
                4. <strong>Classification</strong> → XGBoost / RF / DT
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🏷️ Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)
        
        for cls in CLASS_MAP:
            color = CLASS_COLORS[cls]
            desc = CLASS_DESCRIPTIONS[cls]
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; padding: 8px 12px; background: #f8fafc; border-radius: 8px; border-left: 3px solid {color};">
                <span style="font-weight: 600; color: {color};">{cls}</span>
                <span style="font-size: 0.8rem; color: #64748b;">— {desc}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Distribution visualization and Data Tables
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Distribution chart
        fig_dist = generate_distribution_plot()
        st.pyplot(fig_dist)
        plt.close(fig_dist)
    
    with col2:
        # Data class table
        st.markdown("**Tabel Data Setiap Kelas**")
        data_table = generate_data_class_table()
        st.dataframe(data_table, use_container_width=True, hide_index=True)
    
    # Top Features Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧬 Top Features")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Feature Importance (Top 10)**")
        top_features_df = generate_top_features_table()
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_feat = generate_feature_importance_plot()
        st.pyplot(fig_feat)
        plt.close(fig_feat)


# === PAGE: UPLOAD & PREVIEW ===
def render_upload_page():
    st.markdown("""
    <div class="hero">
        <h1>📤 Upload & Preview</h1>
        <p>Upload a corn leaf image to analyze. Supports JPG, JPEG, and PNG formats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="main_uploader")
        
        # Sample selector
        sample_dir = get_sample_images_path()
        sample_files = ["None"]
        if os.path.exists(sample_dir):
            sample_files += [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(sample_files) > 1:
            selected_sample = st.selectbox("Or use a sample image", sample_files)
        else:
            selected_sample = "None"
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">💡 Tips</div>
            <ul style="color: var(--text-secondary); font-size: 0.85rem; padding-left: 1rem;">
                <li>Use clear, focused images</li>
                <li>Single leaf per image</li>
                <li>Good lighting preferred</li>
                <li>Avoid blurry photos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process image
    image_to_process = None
    source_type = "Upload"
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        st.session_state.image_name = uploaded_file.name
    elif selected_sample != "None":
        sample_path = os.path.join(sample_dir, selected_sample)
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path)
            source_type = "Sample"
            st.session_state.image_name = selected_sample
    
    if image_to_process:
        st.session_state.uploaded_image = image_to_process
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        from modules.preprocessing import preprocess_pil_image
        from modules.segmentation import segment_otsu
        
        img_rgb_norm, gray = preprocess_pil_image(image_to_process)
        segmentation = segment_otsu(gray)
        
        # Three column preview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">Original</div></div>', unsafe_allow_html=True)
            st.image(image_to_process, use_container_width=True)
        
        with col2:
            st.markdown('<div class="card"><div class="card-title">Grayscale</div></div>', unsafe_allow_html=True)
            st.image(gray, use_container_width=True, clamp=True, channels="GRAY")
        
        with col3:
            st.markdown('<div class="card"><div class="card-title">Segmentation</div></div>', unsafe_allow_html=True)
            st.image(segmentation, use_container_width=True, clamp=True, channels="GRAY")
        
        # Image info
        w, h = image_to_process.size
        st.success(f"✅ Image loaded: {w}×{h} pixels | Source: {source_type} | Ready for prediction!")
    else:
        st.info("👆 Upload an image or select a sample to begin analysis.")


# === PAGE: RUN MODEL ===
def render_run_model_page():
    st.markdown("""
    <div class="hero">
        <h1>🔬 Run Model</h1>
        <p>Select a model and run prediction on your uploaded image.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is None:
        st.warning("⚠️ Please upload an image first on the **Upload & Preview** page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">🖼️ Input Image</div></div>', unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, use_container_width=True)
        
        model_choice = st.selectbox("🤖 Select Model", get_available_models())
        
        predict_btn = st.button("🔮 Run Prediction", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn:
            with st.spinner(f"Running {model_choice}..."):
                time.sleep(0.3)
                try:
                    pred_class, probabilities, _ = predict_image_with_model(
                        st.session_state.uploaded_image, model_choice
                    )
                    
                    st.session_state.prediction_result = {
                        'class': pred_class,
                        'probabilities': probabilities,
                        'model': model_choice
                    }
                    
                    st.session_state.predictions_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image': getattr(st.session_state, 'image_name', 'Unknown'),
                        'model': model_choice,
                        'prediction': pred_class,
                        'confidence': float(np.max(probabilities) * 100)
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
        
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            pred_class = result['class']
            probabilities = result['probabilities']
            confidence = np.max(probabilities) * 100
            
            icons = {"Daun Sehat": "🌿", "Daun Rusak": "🍂", "Hawar Daun": "🦠", "Karat Daun": "🟤"}
            
            st.markdown(f"""
            <div class="result-box">
                <div class="result-icon">{icons.get(pred_class, "🍃")}</div>
                <div class="result-class">{pred_class}</div>
                <div class="result-confidence">{confidence:.1f}% confidence</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">Model: {result['model']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability bars
            st.markdown('<div class="card"><div class="card-title">📊 Probability Distribution</div></div>', unsafe_allow_html=True)
            
            class_names = get_class_names()
            for cls, prob in zip(class_names, probabilities):
                color = CLASS_COLORS[cls]
                pct = prob * 100
                st.markdown(f"""
                <div class="prob-bar">
                    <div class="prob-label">
                        <span style="font-weight: {'600' if cls == pred_class else '400'};">{cls}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: {pct}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Metrics section
    if st.session_state.prediction_result:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Model Performance")
        
        model_name = st.session_state.prediction_result['model']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate confusion matrix
            fig_cm = generate_confusion_matrix(model_name)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        
        with col2:
            # Evaluation metrics table for selected model
            st.markdown(f"**Classification Report - {model_name.replace(' (Best)', '')}**")
            eval_df = generate_evaluation_metrics_table(model_name)
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        
        # Model Comparison Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Perbandingan Akurasi Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model comparison chart
            fig_comp = generate_model_comparison_chart()
            st.pyplot(fig_comp)
            plt.close(fig_comp)
        
        with col2:
            # Model comparison table
            st.markdown("**Tabel Perbandingan Akurasi**")
            comparison_df = generate_model_comparison_table()
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# === PAGE: EXPLAINABILITY ===
def render_explainability_page():
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Explainability</h1>
        <p>SHAP analysis reveals how the model makes predictions based on feature contributions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SHAP Summary Plot (Beeswarm)
    st.markdown("### 📈 SHAP Summary Plot")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_summary = generate_shap_summary_plot()
        st.pyplot(fig_summary)
        plt.close(fig_summary)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📖 Cara Membaca</div>
            <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
                Setiap titik = satu sampel. Warna menunjukkan nilai fitur (merah = tinggi, biru = rendah).
                Posisi horizontal menunjukkan dampak terhadap prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Importance Bar Chart
    st.markdown("### 🎯 Feature Importance (Top 10)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_importance = generate_feature_importance_plot()
        st.pyplot(fig_importance)
        plt.close(fig_importance)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧬 Feature Groups</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                <strong>Fine LBP (256)</strong> — Local texture patterns<br>
                <strong>Coarse (32)</strong> — Gradient magnitude<br>
                <strong>DOR (25)</strong> — Dominant regions
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top features table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Top Features Table**")
        top_feat_df = generate_top_features_table()
        st.dataframe(top_feat_df.head(5), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Force Plot & Dependence
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💥 Force Plot")
        fig_force = generate_shap_force_plot()
        st.pyplot(fig_force)
        plt.close(fig_force)
        st.markdown("""
        <p style="font-size: 0.8rem; color: var(--text-secondary);">
            Merah = mendorong prediksi lebih tinggi, Biru = mendorong lebih rendah. Base value adalah rata-rata output model.
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔗 Dependence Plot")
        fig_dep = generate_shap_dependence_plot("Global")
        st.pyplot(fig_dep)
        plt.close(fig_dep)
        st.markdown("""
        <p style="font-size: 0.8rem; color: var(--text-secondary);">
            Menunjukkan hubungan nilai fitur dengan SHAP value. Warna menunjukkan interaksi dengan fitur lain.
        </p>
        """, unsafe_allow_html=True)


# === PAGE: DOWNLOAD ===
def render_download_page():
    st.markdown("""
    <div class="hero">
        <h1>📥 Download & Export</h1>
        <p>Export your prediction history and access visualization gallery.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction History
    st.markdown("### 📋 Prediction History")
    
    if st.session_state.predictions_history:
        df = pd.DataFrame(st.session_state.predictions_history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.predictions_history = []
                st.rerun()
    else:
        st.info("No predictions yet. Run predictions on the **Run Model** page first.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization Gallery - All visualizations organized
    st.markdown("### 🖼️ Visualization Gallery")
    
    # Section 1: Confusion Matrices with Evaluation Metrics
    st.markdown("#### 📊 Confusion Matrices & Evaluation")
    
    for model in ["XGBoost", "Random Forest", "Decision Tree"]:
        st.markdown(f"**{model}**")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = generate_confusion_matrix(model)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            eval_df = generate_evaluation_metrics_table(model)
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 2: Model Comparison
    st.markdown("#### 📈 Perbandingan Akurasi Model")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = generate_model_comparison_chart()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("**Tabel Akurasi**")
        comparison_df = generate_model_comparison_table()
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 3: Dataset Distribution
    st.markdown("#### 📊 Distribusi Dataset")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = generate_distribution_plot()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("**Tabel Data per Kelas**")
        data_df = generate_data_class_table()
        st.dataframe(data_df, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 4: SHAP Visualizations
    st.markdown("#### 🎯 SHAP Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**SHAP Summary Plot**")
        fig = generate_shap_summary_plot()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("**Feature Importance**")
        fig = generate_feature_importance_plot()
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Force Plot**")
        fig = generate_shap_force_plot()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("**Dependence Plot**")
        fig = generate_shap_dependence_plot("Global")
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 5: Top Features Table
    st.markdown("#### 🧬 Top Features")
    top_df = generate_top_features_table()
    st.dataframe(top_df, use_container_width=True, hide_index=True)


# === SIDEBAR ===
with st.sidebar:
    st.markdown("## 🌽 CornShield")
    st.caption("v2.0 | Multi-Model Support")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload & Preview", "🔬 Run Model", "📊 Explainability", "📥 Download"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick legend
    st.markdown("##### 🏷️ Classes")
    for cls in CLASS_MAP:
        color = CLASS_COLORS[cls]
        st.markdown(f'<span style="color: {color}; font-weight: 500;">● {cls}</span>', unsafe_allow_html=True)


# === ROUTING ===
if page == "🏠 Home":
    render_home_page()
elif page == "📤 Upload & Preview":
    render_upload_page()
elif page == "🔬 Run Model":
    render_run_model_page()
elif page == "📊 Explainability":
    render_explainability_page()
elif page == "📥 Download":
    render_download_page()


# === FOOTER ===
st.markdown("""
<div class="footer">
    <strong>CornShield ML</strong> — Machine Learning Final Project 2025
</div>
""", unsafe_allow_html=True)
