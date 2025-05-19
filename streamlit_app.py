import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Face Mask Classification Dashboard", layout="wide")
st.title("😷 Face Mask Detection – Interactive Dashboard")
st.markdown("This dashboard visualizes model training, performance, and explainability for face mask detection using MobileNetV2 and a Custom CNN.")

# Sidebar navigation
section = st.sidebar.radio("Choose Section", [
    "Model Accuracy", "Confusion Matrix", "Grad-CAM Viewer", "Model Comparison", "Classification Report"
])

# Section 1: Accuracy Plot
if section == "Model Accuracy":
    st.subheader("📈 Training & Validation Accuracy")
    col1, col2 = st.columns(2)

    with col1:
        st.image("accuracy_plot.png", caption="MobileNetV2 Accuracy", use_container_width=True)

    with col2:
        st.image("simple_cnn_accuracy.png", caption="Custom CNN Accuracy", use_container_width=True)

# Section 2: Confusion Matrix
elif section == "Confusion Matrix":
    st.subheader("📊 Confusion Matrix")
    st.image("confusion_matrix.png", caption="Confusion Matrix (MobileNetV2)", use_container_width=True)

# Section 3: Grad-CAM Viewer
elif section == "Grad-CAM Viewer":
    st.subheader("🧠 Grad-CAM Comparison")
    model = st.radio("Select Model:", ["MobileNetV2", "Custom CNN"], horizontal=True)

    if model == "MobileNetV2":
        with_mask_img = "gradcam_with_mask.jpg"
        without_mask_img = "gradcam_without_mask.jpg"
    else:
        with_mask_img = "cnn_gradcam_with_mask.jpg"
        without_mask_img = "cnn_gradcam_without_mask.jpg"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("With Mask")
        st.image(with_mask_img, use_container_width=True)

    with col2:
        st.subheader("Without Mask")
        st.image(without_mask_img, use_container_width=True)

# Section 4: Model Comparison Table
elif section == "Model Comparison":
    st.subheader("🤖 Model Performance Comparison")

    st.markdown("""
    | Metric               | MobileNetV2 | Custom CNN |
    |----------------------|-------------|------------|
    | Validation Accuracy  | 95.01%      | 97.16%     |
    | Train Accuracy       | 97.51%      | 97.90%     |
    | Parameters           | ~2.2M       | ~11.1M     |
    | Training Time        | ~15 min     | ~25 min    |
    | Interpretability     | ✅ Good     | ✅ Good     |
    """)

    st.markdown(
        "> The custom CNN achieved higher validation accuracy but is heavier and slower to train. MobileNetV2 remains a strong choice for real-time, lightweight deployment."
    )

# Section 5: Classification Report
elif section == "Classification Report":
    st.subheader("📋 Classification Report – Interactive View")

    # Metrics
    labels = ["with_mask", "without_mask"]
    precision = [0.87, 1.00]
    recall = [1.00, 0.94]
    f1_score = [0.93, 0.97]
    support = [290, 732]

    # Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Precision', x=labels, y=precision, marker_color='mediumturquoise'))
    fig.add_trace(go.Bar(name='Recall', x=labels, y=recall, marker_color='orange'))
    fig.add_trace(go.Bar(name='F1-Score', x=labels, y=f1_score, marker_color='tomato'))

    fig.update_layout(
        title='MobileNetV2 Classification Report',
        yaxis=dict(title='Score'),
        barmode='group',
        xaxis_tickangle=-15
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add support count separately
    st.markdown("**Support:**")
    for label, val in zip(labels, support):
        st.markdown(f"- `{label}`: {val} samples")

    st.markdown("**Accuracy: 0.96**")

st.markdown("---")
st.caption("Interactive dashboard by Ahmed Hujairi | Northumbria University | Powered by Streamlit")
