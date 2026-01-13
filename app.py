import streamlit as st
import numpy as np
from PIL import Image
import os

# Import modules
from modules.pipeline import predict_image, get_class_names
from modules.utils import CLASS_MAP, CLASS_COLORS, CLASS_DESCRIPTIONS

# Page config
st.set_page_config(
    page_title="Corn Leaf Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main .block-container {
        padding-bottom: 100px !important;
        max-width: 1200px;
    }
    
    .stApp {
        background: linear-gradient(to bottom, #f8fafc, #ffffff);
    }
    
    .main-header {
        background: linear-gradient(135deg, #166534 0%, #22c55e 50%, #facc15 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(135deg, #166534 0%, #22c55e 100%);
        color: white;
        text-align: center;
        padding: 1.2rem 2rem;
        font-size: 0.95rem;
        box-shadow: 0 -6px 25px rgba(0, 0, 0, 0.15);
    }
    
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    
    .result-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 3px solid #22c55e;
        margin: 1rem 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌽 Corn Leaf Disease Classifier</h1>
    <p>Klasifikasi Penyakit Daun Jagung dengan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Informasi Model")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #facc15;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    ">
        <strong>🧠 Model:</strong> XGBoost Classifier<br>
        <strong>📊 Fitur:</strong> 313 dimensi<br>
        <strong>🎯 Kelas:</strong> 4 kategori
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 Pipeline Inferensi")
    st.markdown("""
    1. **Preprocessing** - Resize, RGB, Normalisasi, Grayscale
    2. **Segmentasi** - Otsu Thresholding
    3. **Ekstraksi Fitur**:
       - Fine (LBP): 256 dimensi
       - Coarse (Gradient): 32 dimensi
       - DOR: 25 dimensi
    4. **Prediksi** - XGBoost Classification
    """)
    
    st.markdown("---")
    st.markdown("### 🏷️ Kategori Kelas")
    
    for cls_name in CLASS_MAP:
        color = CLASS_COLORS.get(cls_name, "#666")
        st.markdown(f"""
        <div style="
            background: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 4px solid {color};
            font-size: 0.9rem;
        ">
            <strong>{cls_name}</strong>
        </div>
        """, unsafe_allow_html=True)

# Main content
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    padding: 2rem;
    border-radius: 20px;
    border: 3px dashed #22c55e;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.1);
">
    <h3 style="text-align: center; color: #166534; margin-bottom: 1.5rem; font-size: 1.5rem;">
        📤 Upload Gambar Daun Jagung
    </h3>
</div>
""", unsafe_allow_html=True)

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Pilih gambar daun jagung (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload gambar daun jagung untuk klasifikasi penyakit",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File berhasil diupload: {uploaded_file.name}")

with col_sample:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #facc15;
        margin-bottom: 1rem;
    ">
        <h4 style="margin: 0 0 1rem 0; color: #92400e;">📁 Contoh Gambar</h4>
    </div>
    """, unsafe_allow_html=True)
    
    sample_dir = os.path.join(os.path.dirname(__file__), "assets", "sample_images")
    
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if sample_files:
            selected_sample = st.selectbox(
                "Gunakan contoh gambar:",
                ["-- Pilih --"] + sample_files
            )
        else:
            selected_sample = "-- Pilih --"
            st.info("Tidak ada contoh gambar")
    else:
        selected_sample = "-- Pilih --"
        st.info("Folder contoh tidak ditemukan")

# Determine which image to use
image_to_process = None

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
elif selected_sample != "-- Pilih --":
    sample_path = os.path.join(sample_dir, selected_sample)
    if os.path.exists(sample_path):
        image_to_process = Image.open(sample_path)

# Process and display results
if image_to_process is not None:
    
    st.markdown("---")
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("**📷 Gambar Original**")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image_to_process, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Run prediction
    with st.spinner("🔄 Memproses gambar..."):
        try:
            pred_class, probabilities, segmentation = predict_image(image_to_process)
            
            with col2:
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown("**🔍 Hasil Segmentasi Otsu**")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(segmentation, use_container_width=True, clamp=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Show grayscale version
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown("**🔲 Grayscale**")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                # Convert to grayscale for display
                import cv2
                img_array = np.array(image_to_process)
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                st.image(gray_img, use_container_width=True, clamp=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Results section
            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")
            
            col_pred, col_prob = st.columns([1, 2])
            
            with col_pred:
                # Prediction box
                pred_color = CLASS_COLORS.get(pred_class, "#22c55e")
                confidence = probabilities.max() * 100
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">
                        {"🌿" if pred_class == "Daun Sehat" else "🍂"}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {pred_color};">
                        {pred_class}
                    </div>
                    <div style="font-size: 1.1rem; color: #666; margin-top: 0.5rem;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Description
                description = CLASS_DESCRIPTIONS.get(pred_class, "")
                st.info(f"ℹ️ {description}")
            
            with col_prob:
                st.markdown("**📊 Probabilitas Semua Kelas**")
                
                # Create probability bars using Streamlit components
                class_names = get_class_names()
                
                for i, (cls_name, prob) in enumerate(zip(class_names, probabilities)):
                    pct = prob * 100
                    st.write(f"**{cls_name}**: {pct:.1f}%")
                    st.progress(float(prob))
                
        except Exception as e:
            st.error(f"❌ Error saat memproses gambar: {str(e)}")
            st.exception(e)

else:
    # Show placeholder when no image
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 3px dashed #22c55e;
        text-align: center;
        margin: 2rem 0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🌽</div>
        <div style="font-size: 1.5rem; color: #166534; font-weight: 700; margin-bottom: 1rem;">
            Upload Gambar Daun Jagung untuk Memulai Klasifikasi
        </div>
        <div style="font-size: 1rem; color: #666;">
            Sistem akan mendeteksi penyakit berdasarkan tekstur daun menggunakan AI
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-footer">
    <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">🌽</span>
            <strong>Corn Leaf Disease Classifier</strong>
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            Tugas Besar Machine Learning | Klasifikasi Penyakit Daun Jagung
        </div>
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.8;">
        Menggunakan XGBoost dengan fitur Fine, Coarse, dan DOR
    </div>
</div>
""", unsafe_allow_html=True)