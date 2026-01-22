# app_gradcam.py
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import io
from datetime import datetime
import pandas as pd

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Malaria Cell Classifier with Grad-CAM",
    page_icon="🧫",
    layout="wide"
)

# Initialize session state for statistics
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.title("🧫 Advanced Malaria Cell Classifier with Grad-CAM")
st.write("Upload blood cell images to predict their class and visualize model attention using Grad-CAM.")

# -----------------------
# Load Model (cached)
# -----------------------
@st.cache_resource
def load_model(model_path="resnet18_malaria_finetuned_best.pth"):
    """Load the trained ResNet18 model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Please ensure 'resnet18_malaria_finetuned_best.pth' is in the same directory as this script.")
    model_loaded = False
    st.stop()

# -----------------------
# Image Preprocessing
# -----------------------
def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# -----------------------
# Prediction Function
# -----------------------
def predict(image: Image.Image):
    """Make prediction on input image"""
    tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    class_names = ['Parasitized', 'Uninfected']
    return class_names[pred_class.item()], confidence.item(), probs.cpu().numpy()[0]

# -----------------------
# Grad-CAM Implementation
# -----------------------
class GradCAM:
    """Generate Grad-CAM heatmap for CNN interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
        
        # Backward pass
        loss = outputs[0, class_idx]
        loss.backward()
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Resize and normalize
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def __del__(self):
        """Clean up hooks"""
        for handle in self.hook_handles:
            handle.remove()

# -----------------------
# Visualization Function
# -----------------------
def create_gradcam_overlay(image, cam, alpha=0.5):
    """Create Grad-CAM overlay on original image"""
    # Resize image to 224x224
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Blend heatmap with image
    overlay = cv2.addWeighted(heatmap, alpha, img_np, 1 - alpha, 0)
    
    return overlay, heatmap

# -----------------------
# Sidebar Configuration
# -----------------------
st.sidebar.header("⚙️ Settings")

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help="Predictions below this threshold will be flagged for review"
)

# Batch processing option
batch_mode = st.sidebar.checkbox(
    "Batch Processing Mode",
    value=False,
    help="Upload and analyze multiple images at once"
)

# Grad-CAM opacity
gradcam_alpha = st.sidebar.slider(
    "Grad-CAM Overlay Opacity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Adjust the transparency of the heatmap overlay"
)

st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a fine-tuned ResNet18 model to classify blood cell images as:
- **🦠 Parasitized**: Contains malaria parasites
- **✅ Uninfected**: Healthy cells

**Grad-CAM** visualizes which image regions influenced the model's decision.
""")

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"""
- Architecture: ResNet18
- Input size: 224×224
- Classes: 2
- Test Accuracy: 97.22%
- Device: {device}
""")

# Educational content
with st.sidebar.expander("🔬 What to Look For"):
    st.write("""
    **Parasitized Cells:**
    - Contain visible parasites (dark spots/rings)
    - May show ring stage or trophozoite
    - Irregular cell membrane
    - Altered cell color/texture
    
    **Uninfected Cells:**
    - Uniform red/pink color
    - Smooth, regular shape
    - No internal parasites
    - Clear cell membrane
    """)

# -----------------------
# Main Application
# -----------------------
if not batch_mode:
    # Single image mode
    st.header("📤 Single Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["png", "jpg", "jpeg"],
        help="Upload a blood cell microscopy image"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            label, confidence, probs = predict(image)
            
            # Add to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'filename': uploaded_file.name,
                'prediction': label,
                'confidence': confidence,
                'parasitized_prob': probs[0],
                'uninfected_prob': probs[1]
            })
        
        # Display prediction results
        st.subheader("📋 Prediction Results")
        
        # Confidence warning
        if confidence < confidence_threshold:
            st.warning(f"⚠️ Low confidence prediction ({confidence*100:.2f}%). Consider manual review.")
        
        # Create metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Predicted Class", label)
        with metric_col2:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        with metric_col3:
            status = "✅ High Confidence" if confidence >= confidence_threshold else "⚠️ Review Needed"
            st.metric("Status", status)
        
        # Show probabilities
        st.write("**Class Probabilities:**")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.progress(float(probs[0]), text=f"🦠 Parasitized: {float(probs[0])*100:.2f}%")
        with prob_col2:
            st.progress(float(probs[1]), text=f"✅ Uninfected: {float(probs[1])*100:.2f}%")
        
        # Generate Grad-CAM
        st.subheader("🔍 Grad-CAM Visualization")
        
        with st.spinner("Generating Grad-CAM..."):
            gradcam = GradCAM(model, target_layer=model.layer4[-1])
            input_tensor = preprocess_image(image).to(device)
            cam = gradcam.generate(input_tensor)
            
            # Create visualizations
            overlay, heatmap = create_gradcam_overlay(image, cam, alpha=gradcam_alpha)
        
        # Display Grad-CAM results
        with col2:
            st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
        with col3:
            st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
        
        st.info("""
        🔍 **Interpretation**: Warmer colors (red/yellow) indicate areas the model focused on most. 
        For parasitized cells, the model typically focuses on the parasite itself. For uninfected cells, 
        it may focus on the cell membrane and uniform interior.
        """)
        
        # Download buttons
        st.subheader("📥 Downloads")
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            overlay_pil = Image.fromarray(overlay)
            buf1 = io.BytesIO()
            overlay_pil.save(buf1, format='PNG')
            st.download_button(
                label="Download Grad-CAM Overlay",
                data=buf1.getvalue(),
                file_name=f"gradcam_{label.lower()}_{confidence*100:.0f}pct.png",
                mime="image/png"
            )
        
        with dl_col2:
            heatmap_pil = Image.fromarray(heatmap)
            buf2 = io.BytesIO()
            heatmap_pil.save(buf2, format='PNG')
            st.download_button(
                label="Download Heatmap Only",
                data=buf2.getvalue(),
                file_name=f"heatmap_{label.lower()}_{confidence*100:.0f}pct.png",
                mime="image/png"
            )

else:
    # Batch processing mode
    st.header("📤 Batch Processing Mode")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload multiple blood cell microscopy images"
    )
    
    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} images...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_results = []
        
        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({idx+1}/{len(uploaded_files)})")
            
            try:
                image = Image.open(uploaded_file).convert("RGB")
                label, confidence, probs = predict(image)
                
                batch_results.append({
                    'Filename': uploaded_file.name,
                    'Prediction': label,
                    'Confidence': f"{confidence*100:.2f}%",
                    'Parasitized_Prob': f"{probs[0]*100:.2f}%",
                    'Uninfected_Prob': f"{probs[1]*100:.2f}%",
                    'Status': '✅ High Confidence' if confidence >= confidence_threshold else '⚠️ Review',
                    'Image': image,
                    'Confidence_Raw': confidence
                })
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'filename': uploaded_file.name,
                    'prediction': label,
                    'confidence': confidence,
                    'parasitized_prob': probs[0],
                    'uninfected_prob': probs[1]
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        
        # Display results table
        st.subheader("📊 Batch Results Summary")
        
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Image', 'Confidence_Raw']} 
                          for r in batch_results])
        st.dataframe(df, use_column_width=True)
        
        # Statistics
        st.subheader("📈 Batch Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        parasitized_count = sum(1 for r in batch_results if r['Prediction'] == 'Parasitized')
        uninfected_count = sum(1 for r in batch_results if r['Prediction'] == 'Uninfected')
        low_confidence_count = sum(1 for r in batch_results if r['Confidence_Raw'] < confidence_threshold)
        avg_confidence = np.mean([r['Confidence_Raw'] for r in batch_results])
        
        with stat_col1:
            st.metric("Parasitized", parasitized_count)
        with stat_col2:
            st.metric("Uninfected", uninfected_count)
        with stat_col3:
            st.metric("Low Confidence", low_confidence_count)
        with stat_col4:
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        
        # Download batch results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Show individual images with low confidence
        low_conf_images = [r for r in batch_results if r['Confidence_Raw'] < confidence_threshold]
        
        if low_conf_images:
            st.subheader("⚠️ Images Requiring Review")
            st.write(f"Found {len(low_conf_images)} image(s) with confidence below {confidence_threshold*100:.0f}%")
            
            for result in low_conf_images:
                with st.expander(f"{result['Filename']} - {result['Prediction']} ({result['Confidence']})"):
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    
                    with col_a:
                        st.image(result['Image'], caption="Original", use_column_width=True)
                    
                    # Generate Grad-CAM for review
                    gradcam = GradCAM(model, target_layer=model.layer4[-1])
                    input_tensor = preprocess_image(result['Image']).to(device)
                    cam = gradcam.generate(input_tensor)
                    overlay, heatmap = create_gradcam_overlay(result['Image'], cam, alpha=gradcam_alpha)
                    
                    with col_b:
                        st.image(heatmap, caption="Heatmap", use_column_width=True)
                    with col_c:
                        st.image(overlay, caption="Overlay", use_column_width=True)

# -----------------------
# Prediction History
# -----------------------
if st.session_state.prediction_history:
    st.header("📜 Prediction History")
    
    with st.expander(f"View All Predictions ({len(st.session_state.prediction_history)} total)"):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
        history_df['parasitized_prob'] = history_df['parasitized_prob'].apply(lambda x: f"{x*100:.2f}%")
        history_df['uninfected_prob'] = history_df['uninfected_prob'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(history_df, use_container_width=True)
        
        # Overall statistics
        st.subheader("📊 Overall Statistics")
        
        total_predictions = len(st.session_state.prediction_history)
        parasitized_total = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 'Parasitized')
        uninfected_total = total_predictions - parasitized_total
        
        overall_col1, overall_col2, overall_col3 = st.columns(3)
        with overall_col1:
            st.metric("Total Predictions", total_predictions)
        with overall_col2:
            st.metric("Parasitized", f"{parasitized_total} ({parasitized_total/total_predictions*100:.1f}%)")
        with overall_col3:
            st.metric("Uninfected", f"{uninfected_total} ({uninfected_total/total_predictions*100:.1f}%)")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# -----------------------
# Usage Instructions
# -----------------------
if (not batch_mode and uploaded_file is None) or (batch_mode and not uploaded_files):
    st.info("👆 Please upload image(s) to get started!")
    
    with st.expander("📖 How to Use This App"):
        st.write("""
        ### Single Image Mode
        1. **Upload an image**: Click the file uploader and select a blood cell microscopy image
        2. **View prediction**: See the classification result with confidence score
        3. **Examine Grad-CAM**: Understand which parts influenced the model's decision
        4. **Download results**: Save visualizations for your records
        
        ### Batch Processing Mode
        1. **Enable batch mode**: Check the "Batch Processing Mode" in the sidebar
        2. **Upload multiple images**: Select several images at once
        3. **Review results**: See a summary table of all predictions
        4. **Flag low confidence**: Images below the threshold are highlighted for review
        5. **Download CSV**: Export results for further analysis
        
        ### Settings
        - **Confidence Threshold**: Predictions below this value are flagged for manual review
        - **Grad-CAM Opacity**: Adjust the visibility of the heatmap overlay
        
        ### Tips
        - For best results, use clear microscopy images of individual blood cells
        - Images with confidence below 85% should be reviewed by a medical professional
        - The model performs best on images similar to its training data
        """)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns([1, 1, 1])

with col_footer1:
    st.markdown("**Model Performance**")
    st.write("Test Accuracy: 97.22%")

with col_footer2:
    st.markdown("**Technology Stack**")
    st.write("Streamlit • PyTorch • ResNet18")

with col_footer3:
    st.markdown("**Medical Disclaimer**")
    st.write("For research purposes only")

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 20px;'>"
    "⚕️ This tool is for educational and research purposes. Always consult medical professionals for diagnosis."
    "</div>",
    unsafe_allow_html=True
)
