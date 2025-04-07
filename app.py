import streamlit as st
import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import threading
import queue
import shutil
import base64
from datetime import datetime
import tempfile
import io

# Try to import OpenCV with proper error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.error("""
        OpenCV (cv2) is not properly installed. Please ensure you have installed the required dependencies:
        ```
        pip install -r requirements.txt
        ```
        If you're deploying to Streamlit Cloud, make sure to include opencv-python-headless in your requirements.txt file.
    """)
    st.stop()

# Add parent directory to path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.append(current_dir)  # Make sure utils can be imported from current dir

try:
    from utils import LicensePlateWebcamDetector, init_webcam
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.stop()

# Global variables
output_dir = os.path.join(current_dir, "detected_plates")  # Directory to save detections

def get_image_base64(image_path):
    """Convert image to base64 string for HTML display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_image(image_data, detector, use_ocr=False, save_plates=True, output_dir=None):
    """Process image from webcam input"""
    try:
        # Convert to OpenCV format
        if isinstance(image_data, np.ndarray):
            frame = image_data.copy()
        else:
            # Convert from PIL to OpenCV
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            frame = np.array(pil_image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process the frame
        processed_frame, plate_info = detector.process_frame(
            frame,
            use_ocr=use_ocr,
            save_plates=save_plates,
            output_dir=output_dir
        )
        
        # Convert back to RGB for display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        return rgb_frame, plate_info
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, []

def main():
    st.set_page_config(
        page_title="License Plate Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed",  # Collapsed sidebar for mobile
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "# License Plate Detection System\nThis is a mobile-friendly application for detecting license plates."
        }
    )
    
    # Add custom CSS for mobile responsiveness
    st.markdown("""
        <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .stApp {
                padding: 0.5rem;
            }
            .stButton>button {
                width: 100%;
                margin: 0.25rem 0;
                height: 3rem;
                font-size: 1.1rem;
            }
            .stImage {
                max-width: 100%;
                height: auto;
            }
            .stDataFrame {
                font-size: 0.8rem;
                width: 100%;
                overflow-x: auto;
            }
            .stTextInput>div>div>input {
                font-size: 1rem;
                height: 2.5rem;
            }
            .stSelectbox>div>div>select {
                font-size: 1rem;
                height: 2.5rem;
            }
            .stSlider>div>div>div {
                height: 2.5rem;
            }
            .stCheckbox>div>label {
                font-size: 1rem;
            }
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            .stExpander {
                margin-bottom: 1rem;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                font-size: 1.5rem;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
        }
        
        /* Touch-friendly elements */
        .stButton>button, .stSelectbox>div>div>select, .stTextInput>div>div>input {
            touch-action: manipulation;
        }
        
        /* Improved visibility */
        .stAlert {
            margin: 0.5rem 0;
            padding: 0.75rem;
        }
        
        /* Better spacing for mobile */
        .element-container {
            margin-bottom: 0.5rem;
        }
        
        /* Fix for mobile camera display */
        .stImage img {
            max-width: 100%;
            height: auto;
            object-fit: contain;
        }
        
        /* Improve table readability on mobile */
        .stDataFrame td, .stDataFrame th {
            padding: 0.5rem;
            white-space: nowrap;
        }
        
        /* Make camera component more mobile-friendly */
        [data-testid="stCamera"] {
            width: 100% !important;
            margin: 0 auto;
        }
        
        [data-testid="stCamera"] > div {
            width: 100% !important;
        }
        
        [data-testid="stCamera"] video {
            width: 100% !important;
            height: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("License Plate Detection System")
    
    # Initialize session state variables
    if 'detector' not in st.session_state:
        st.session_state['detector'] = None
    if 'detection_running' not in st.session_state:
        st.session_state['detection_running'] = False
    if 'save_plates' not in st.session_state:
        st.session_state['save_plates'] = True
    if 'detections' not in st.session_state:
        st.session_state['detections'] = []
    if 'last_processed_image' not in st.session_state:
        st.session_state['last_processed_image'] = None
    if 'detection_results' not in st.session_state:
        st.session_state['detection_results'] = []
    
    # Move settings to an expander for mobile
    with st.expander("Settings", expanded=False):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
            
        # Model settings
        model_path = None
        local_model = os.path.join(current_dir, "best.pt")
        parent_model = os.path.join(parent_dir, "best.pt")
        
        if os.path.exists(local_model):
            model_path = local_model
        elif os.path.exists(parent_model):
            shutil.copy(parent_model, local_model)
            model_path = local_model
            
        if model_path:
            st.success(f"Using model: {os.path.basename(model_path)}")
        else:
            st.error("No model found. Please upload a model file.")
            uploaded_model = st.file_uploader("Upload YOLO model (.pt)", type=['pt'])
            if uploaded_model:
                with open(local_model, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model_path = local_model
                st.success(f"Model uploaded successfully!")
        
        # Camera settings with mobile-friendly layout
        st.subheader("Camera Settings")
        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        with col2:
            save_plates = st.checkbox("Save Detected Plates", value=st.session_state['save_plates'], key="save_plates_settings")
            if save_plates != st.session_state['save_plates']:
                st.session_state['save_plates'] = save_plates
        
        # Gemini API key for OCR
        st.subheader("OCR Settings")
        gemini_key = st.text_input("Gemini API Key (for OCR)", type="password")
        
        # Device selection
        st.subheader("Performance Settings")
        device_options = ["CPU", "GPU"]
        device_selection = st.selectbox("Device", device_options, index=0)  # Default to CPU for mobile compatibility
        device = "cpu" if device_selection == "CPU" else "0"
    
    # Initialize detector if model is available
    if model_path and st.session_state['detector'] is None:
        with st.spinner("Initializing detector..."):
            try:
                st.session_state['detector'] = LicensePlateWebcamDetector(
                    model_path=model_path,
                    device=device,
                    conf_threshold=conf_threshold,
                    gemini_api_key=gemini_key if gemini_key else None
                )
                st.success("Detector initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing detector: {e}")
    
    # Create main interface
    # Use a single column layout for mobile
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:first-child {
            flex: 1;
        }
        @media (max-width: 768px) {
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Show instructions for mobile users
    st.info("""
        📱 **Mobile Camera Usage:**
        1. Grant camera permissions when prompted
        2. Take a photo or upload an existing one
        3. The system will analyze the image for license plates
    """)
    
    # Use Streamlit's native camera input for mobile compatibility
    camera_tab, upload_tab = st.tabs(["Camera", "Upload Image"])
    
    with camera_tab:
        # Use Streamlit's camera input component which works better on mobile
        camera_image = st.camera_input("Take a photo of a license plate", key="license_plate_camera")
        
        if camera_image is not None:
            # Process the camera image if detector is available
            if st.session_state['detector'] is not None:
                try:
                    with st.spinner("Processing image..."):
                        # Process the image
                        processed_image, results = process_image(
                            camera_image.getvalue(),
                            st.session_state['detector'],
                            use_ocr=True if gemini_key else False,
                            save_plates=st.session_state['save_plates'],
                            output_dir=output_dir
                        )
                        
                        if processed_image is not None:
                            # Save results to session state
                            st.session_state['last_processed_image'] = processed_image
                            st.session_state['detection_results'] = results
                            
                            # Display processed image
                            st.image(
                                processed_image,
                                caption="Processed Image with Detections",
                                use_column_width=True
                            )
                            
                            # Display detection results
                            if results:
                                st.success(f"Detected {len(results)} license plate(s)")
                                
                                # Create a table of results
                                data = []
                                for i, plate in enumerate(results):
                                    data.append({
                                        "Index": i+1,
                                        "Type": plate.get('type', 'Unknown'),
                                        "Plate Text": plate.get('plate_text', 'N/A'),
                                        "Confidence": f"{plate.get('confidence', 0):.2f}"
                                    })
                                
                                if data:
                                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                            else:
                                st.warning("No license plates detected in this image")
                        else:
                            st.error("Failed to process the image")
                except Exception as e:
                    st.error(f"Error processing camera image: {str(e)}")
    
    with upload_tab:
        # Allow upload of images for processing
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Process the uploaded image if detector is available
            if st.session_state['detector'] is not None:
                try:
                    with st.spinner("Processing image..."):
                        # Process the image
                        processed_image, results = process_image(
                            uploaded_image.getvalue(),
                            st.session_state['detector'],
                            use_ocr=True if gemini_key else False,
                            save_plates=st.session_state['save_plates'],
                            output_dir=output_dir
                        )
                        
                        if processed_image is not None:
                            # Save results to session state
                            st.session_state['last_processed_image'] = processed_image
                            st.session_state['detection_results'] = results
                            
                            # Display processed image
                            st.image(
                                processed_image,
                                caption="Processed Image with Detections",
                                use_column_width=True
                            )
                            
                            # Display detection results
                            if results:
                                st.success(f"Detected {len(results)} license plate(s)")
                                
                                # Create a table of results
                                data = []
                                for i, plate in enumerate(results):
                                    data.append({
                                        "Index": i+1,
                                        "Type": plate.get('type', 'Unknown'),
                                        "Plate Text": plate.get('plate_text', 'N/A'),
                                        "Confidence": f"{plate.get('confidence', 0):.2f}"
                                    })
                                
                                if data:
                                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                            else:
                                st.warning("No license plates detected in this image")
                        else:
                            st.error("Failed to process the image")
                except Exception as e:
                    st.error(f"Error processing uploaded image: {str(e)}")
    
    # Display saved detections
    st.subheader("Saved Detections")
    
    # Check for detected_plates directory and files
    if os.path.exists(output_dir):
        plate_images = list(Path(output_dir).glob("*.jpg"))
        plate_images = [p for p in plate_images if "frame_" not in p.name]  # Exclude full frames
        
        if plate_images:
            plate_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Display in grid
            cols = st.columns(3)
            for i, img_path in enumerate(plate_images[:9]):  # Display up to 9 images
                with cols[i % 3]:
                    st.image(str(img_path), caption=img_path.name, use_column_width=True)
        else:
            st.info("No saved detections found.")
    else:
        st.info("No detection directory found.")
    
    # Display detection history from CSV if available
    csv_path = os.path.join(output_dir, "detection_history.csv")
    if os.path.exists(csv_path):
        with st.expander("Detection Records", expanded=False):
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
                
                # Download button for CSV
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Detection Data",
                    data=csv_data,
                    file_name="license_plate_detections.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error loading detection history: {e}")

if __name__ == "__main__":
    main() 