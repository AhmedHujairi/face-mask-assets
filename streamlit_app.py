import streamlit as st
from PIL import Image

st.set_page_config(page_title="Grad-CAM Viewer", layout="wide")
st.title("🧠 Grad-CAM Comparison Viewer")
st.markdown("Interactively compare Grad-CAM visualizations from different models.")

model_choice = st.radio("Choose Model:", ["MobileNetV2", "Custom CNN"])

if model_choice == "MobileNetV2":
    with_mask_img = "gradcam_with_mask.jpg"
    without_mask_img = "gradcam_without_mask.jpg"
else:
    with_mask_img = "gradcam_with_mask.jpg"  # Replace if you had separate CNN Grad-CAMs
    without_mask_img = "gradcam_without_mask.jpg"

col1, col2 = st.columns(2)

with col1:
    st.subheader("With Mask")
    st.image(with_mask_img, use_column_width=True)

with col2:
    st.subheader("Without Mask")
    st.image(without_mask_img, use_column_width=True)

st.caption("Visualizations generated using Grad-CAM. Highlighted regions indicate where the model focused when making predictions.")
