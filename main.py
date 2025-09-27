import streamlit as st
from keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO
import cv2
import plotly.graph_objects as go
from datetime import datetime

from disease_info import DISEASE_DATABASE

st.set_page_config(
    page_title="FarmerAid - Plant Disease Detection",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .disease-card {
        background: #333;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prevention-tip {
        background: #333;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .treatment-box {
        background: #333;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_disease_model():
    try:
        model = load_model('disease.keras')

        # Debug: Print model information
        st.write("Model loaded successfully!")
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
        st.write(f"Expected number of classes: {model.output_shape[-1]}")
        st.write(f"Our class names count: {len(CLASS_NAMES)}")

        # Verify class count matches
        if model.output_shape[-1] != len(CLASS_NAMES):
            st.error(f"⚠️ MODEL MISMATCH: Model expects {model.output_shape[-1]} classes but we have {len(CLASS_NAMES)} class names!")
            st.info("Please verify your class names match the model's training data.")

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure your model file 'disease.keras' is in the correct directory")
        return None

def preprocess_image(img):
    """Enhanced image preprocessing with validation"""
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(img)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        original_shape = img_array.shape

        # Validate image
        if len(original_shape) < 3 or original_shape[2] != 3:
            return None, original_shape, "Image must be RGB (3 channels)"

        # Normalize and resize
        img_normalized = img_array / 255.0
        img_resized = cv2.resize(img_normalized, (128, 128))
        img_batch = np.expand_dims(img_resized, axis=0)

        return img_batch, original_shape, None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def create_prediction_chart(predictions, class_names):
    """Create an interactive prediction confidence chart"""
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_predictions = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    fig = go.Figure(data=[
        go.Bar(
            x=top_classes,
            y=top_predictions * 100,
            marker_color=['#ff6b6b' if i == 0 else '#4ecdc4' for i in range(len(top_classes))],
            text=[f'{p:.1f}%' for p in top_predictions * 100],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Top 5 Disease Prediction Confidences',
        xaxis_title='Disease Type',
        yaxis_title='Confidence (%)',
        showlegend=False,
        height=400,
        font=dict(size=12),
        xaxis_tickangle=-45
    )

    return fig

def get_confidence_class(confidence):
    """Get confidence level classification"""
    if confidence >= 0.8:
        return "confidence-high", "High Confidence", "🟢"
    elif confidence >= 0.6:
        return "confidence-medium", "Medium Confidence", "🟡"
    else:
        return "confidence-low", "Low Confidence", "🔴"

def display_disease_info(disease_name, confidence):
    """Display comprehensive disease information"""
    if disease_name not in DISEASE_DATABASE:
        st.error(f"Disease information not available for: {disease_name}")
        st.info("Available diseases in database:")
        for key in DISEASE_DATABASE.keys():
            st.write(f"- {key}")
        return

    disease_info = DISEASE_DATABASE[disease_name]
    conf_class, conf_text, conf_emoji = get_confidence_class(confidence)

    # Main prediction result
    st.markdown(f"""
    <div class="disease-card">
        <h2>🔬 Diagnosis: {disease_name}</h2>
        <p><strong>Scientific Name:</strong> {disease_info['scientific_name']}</p>
        <p><strong>Severity Level:</strong> <span style="color: {'red' if disease_info['severity'] == 'Very High' else 'orange' if disease_info['severity'] == 'High' else 'green'}">{disease_info['severity']}</span></p>
        <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.1%} ({conf_text}) {conf_emoji}</span></p>
        <p><strong>Action Required:</strong> {disease_info['urgency']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Disease description
    st.markdown("### 📋 Disease Description")
    st.info(disease_info['description'])

    # Create tabs for different information sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Symptoms", "💊 Treatment", "🛡️ Prevention", "🌱 Organic Solutions"])

    with tab1:
        st.markdown("#### Symptoms to Look For:")
        for symptom in disease_info['symptoms']:
            st.markdown(f"• {symptom}")

    with tab2:
        st.markdown("#### Treatment Recommendations:")
        for treatment in disease_info['treatment']:
            st.markdown(f"""
            <div class="treatment-box">
                • {treatment}
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Prevention Strategies:")
        for prevention in disease_info['prevention']:
            st.markdown(f"""
            <div class="prevention-tip">
                • {prevention}
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("#### Organic & Natural Solutions:")
        for solution in disease_info['organic_solutions']:
            st.markdown(f"🌿 {solution}")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍅 FarmerAid - Advanced Plant Disease Detection</h1>
        <p>Protecting your crops with AI-powered disease identification</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Settings & Info")

        # Model info
        st.info(f"""
        **Model Classes:** {len(CLASS_NAMES)}
        **Supported Plants:** Tomato, Potato, Pepper
        **Image Requirements:** RGB, any size
        **Processing Time:** ~2 seconds
        """)

        # Debug mode
        debug_mode = st.checkbox("Enable Debug Mode", value=True)

        # Image enhancement options
        st.markdown("### 📸 Image Enhancement")
        enhance_image = st.checkbox("Enable image enhancement", value=False)

        if enhance_image:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Take photos in good lighting
        - Focus on affected leaf areas
        - Avoid blurry or dark images
        - Multiple angles help accuracy
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Plant Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a plant leaf showing any symptoms"
        )

        if uploaded_file:
            # Display original image
            image = Image.open(uploaded_file)

            # Apply enhancements if enabled
            if enhance_image:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)

            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Image info
            st.markdown(f"""
            **Image Info:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)

    with col2:
        if uploaded_file:
            st.markdown("### 🔍 Analysis Results")

            # Load model
            model = load_disease_model()
            if model is None:
                st.error("Model not available. Please check the model file.")
                return

            # Process image
            with st.spinner("🔬 Analyzing image..."):
                processed_img, original_shape, error = preprocess_image(uploaded_file)

                if error:
                    st.error(error)
                    return

                # Make prediction
                predictions = model.predict(processed_img)[0]
                predicted_class_idx = np.argmax(predictions)
                confidence = np.max(predictions)

                if debug_mode:
                    st.write(f"**Debug Info:**")
                    st.write(f"Predicted class index: {predicted_class_idx}")
                    st.write(f"Total classes in model output: {len(predictions)}")
                    st.write(f"Raw predictions (first 5): {predictions[:5]}")
                    st.write(f"Top 3 predictions:")
                    top_3_indices = np.argsort(predictions)[-3:][::-1]
                    for i, idx in enumerate(top_3_indices):
                        st.write(f"{i+1}. {CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else 'Unknown'}: {predictions[idx]:.3f}")

                # Validate prediction index
                if predicted_class_idx >= len(CLASS_NAMES):
                    st.error(f"Prediction index {predicted_class_idx} is out of range for class names list!")
                    return

                predicted_disease = CLASS_NAMES[predicted_class_idx]

            # Display results
            st.success("✅ Analysis Complete!")

            # Prediction confidence chart
            fig = create_prediction_chart(predictions, CLASS_NAMES)
            st.plotly_chart(fig, use_container_width=True)

            # Quick metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Top Prediction", predicted_disease.replace('Tomato ', '').replace('Potato ', '').replace('Pepper Bell ', ''))
            with col_b:
                st.metric("Confidence", f"{confidence:.1%}")
            with col_c:
                if predicted_disease in DISEASE_DATABASE:
                    urgency = DISEASE_DATABASE[predicted_disease]['urgency'].split(' - ')[0]
                    st.metric("Urgency", urgency)
                else:
                    st.metric("Urgency", "Unknown")

    # Detailed results section
    if uploaded_file and 'predicted_disease' in locals():
        st.markdown("---")
        if predicted_disease in DISEASE_DATABASE:
            display_disease_info(predicted_disease, confidence)
        else:
            st.warning(f"No detailed information available for: {predicted_disease}")
            st.info("Please ensure your disease_info.py contains information for all predicted classes.")

        st.markdown("### 📋 Recommended Action Plan")

        if confidence >= 0.7:
            st.success("High confidence prediction - Follow treatment recommendations")
        elif confidence >= 0.5:
            st.warning("Moderate confidence - Consider getting a second opinion or additional photos")
        else:
            st.error("Low confidence prediction - Retake photo with better lighting/focus")

        # Download report
        if st.button("📄 Generate Disease Report"):
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_disease': predicted_disease,
                'confidence': f"{confidence:.1%}",
                'urgency': DISEASE_DATABASE[predicted_disease]['urgency'] if predicted_disease in DISEASE_DATABASE else "Unknown",
                'treatment_summary': DISEASE_DATABASE[predicted_disease]['treatment'][:3] if predicted_disease in DISEASE_DATABASE else ["No information available"]
            }

            st.success("📄 Report generated successfully!")
            st.json(report_data)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #333; border-radius: 10px;'>
        <h3>🌱 About FarmerAid</h3>
        <p>FarmerAid uses advanced machine learning to help farmers identify and treat plant diseases early,
        protecting crops and improving yields. Our AI model is trained on thousands of plant images to provide
        accurate, actionable insights.</p>
        <p><strong>Disclaimer:</strong> This tool provides AI-based suggestions. For severe cases or uncertainties,
        consult with agricultural extension services or plant pathologists.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()