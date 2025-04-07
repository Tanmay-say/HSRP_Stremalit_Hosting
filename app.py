import streamlit as st
import cv2
import os
import time
import numpy as np
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import threading
import queue
import shutil
import base64
from datetime import datetime
import tempfile

# Add parent directory to path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.append(current_dir)  # Make sure utils can be imported from current dir

from utils import LicensePlateWebcamDetector, init_webcam

# Global variables
frame_queue = queue.Queue(maxsize=2)  # Queue to hold frames
stop_event = threading.Event()  # Event to signal thread to stop
processed_frame = None  # Most recent processed frame
detection_data = []  # Store detection data
output_dir = os.path.join(current_dir, "detected_plates")  # Directory to save detections

def webcam_thread(camera_id, width, height):
    """Thread to continuously read frames from webcam"""
    # Initialize webcam with retry mechanism
    max_retries = 3
    retry_count = 0
    cap = None
    
    while retry_count < max_retries:
        try:
            cap = init_webcam(camera_id, width, height)
            if cap is not None and cap.isOpened():
                break
            retry_count += 1
            time.sleep(1)  # Wait before retrying
        except Exception as e:
            st.error(f"Camera initialization error: {str(e)}")
            retry_count += 1
            time.sleep(1)
    
    if cap is None or not cap.isOpened():
        st.error(f"Error: Could not open camera {camera_id} after {max_retries} attempts")
        return
    
    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from camera. Attempting to reconnect...")
                cap.release()
                cap = init_webcam(camera_id, width, height)
                if not cap.isOpened():
                    st.error("Could not reconnect to camera")
                    break
                continue
                
            # Resize frame for mobile optimization
            frame = cv2.resize(frame, (width, height))
                
            # If queue is full, remove oldest frame
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            # Add new frame to queue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
                
        except Exception as e:
            st.error(f"Error in webcam thread: {str(e)}")
            break
            
    # Release resources
    if cap is not None:
        cap.release()

def get_image_base64(image_path):
    """Convert image to base64 string for HTML display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

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
        </style>
    """, unsafe_allow_html=True)
    
    st.title("License Plate Detection System")
    
    # Move settings to an expander for mobile
    with st.expander("Settings", expanded=False):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detector settings in session state if not exist
        if 'detector' not in st.session_state:
            st.session_state.detector = None
        if 'detection_running' not in st.session_state:
            st.session_state.detection_running = False
        if 'save_plates' not in st.session_state:
            st.session_state.save_plates = True
        if 'detections' not in st.session_state:
            st.session_state.detections = []
            
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
            camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
            cam_width = st.number_input("Camera Width", min_value=320, max_value=1920, value=640)
        with col2:
            cam_height = st.number_input("Camera Height", min_value=240, max_value=1080, value=480)
            conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        
        # Gemini API key for OCR
        st.subheader("OCR Settings")
        gemini_key = st.text_input("Gemini API Key (for OCR)", type="password")
        
        # Device selection
        st.subheader("Performance Settings")
        device_options = ["CPU", "GPU"]
        device_selection = st.selectbox("Device", device_options, index=1)
        device = "cpu" if device_selection == "CPU" else "0"
    
    # Initialize detector if model is available
    if model_path and st.session_state.detector is None:
        with st.spinner("Initializing detector..."):
            try:
                st.session_state.detector = LicensePlateWebcamDetector(
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
    
    # Webcam feed display with responsive container
    st.markdown('<div style="width: 100%; max-width: 100%; margin: 0 auto;">', unsafe_allow_html=True)
    st.subheader("Live Detection")
    video_placeholder = st.empty()
    status_text = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Start/Stop detection buttons with mobile-friendly layout
    detection_col1, detection_col2 = st.columns([1, 1])
    with detection_col1:
        if not st.session_state.detection_running:
            if st.button("Start Detection", use_container_width=True, key="start_detection"):
                # Start webcam thread
                if st.session_state.detector is not None:
                    stop_event.clear()
                    thread = threading.Thread(
                        target=webcam_thread,
                        args=(camera_id, cam_width, cam_height),
                        daemon=True
                    )
                    thread.start()
                    st.session_state.detection_running = True
                    st.rerun()
        else:
            if st.button("Stop Detection", use_container_width=True, key="stop_detection"):
                # Stop webcam thread
                stop_event.set()
                st.session_state.detection_running = False
                st.rerun()
    
    with detection_col2:
        # Toggle plate saving with mobile-friendly layout
        save_plates = st.checkbox("Save Detected Plates", value=st.session_state.save_plates, key="save_plates")
        st.session_state.save_plates = save_plates
    
    # Detection history with mobile-friendly layout
    st.markdown("""
        <style>
        .detection-history {
            margin-top: 1rem;
            padding: 0.5rem;
            border-radius: 0.5rem;
            background-color: rgba(0,0,0,0.05);
        }
        @media (max-width: 768px) {
            .detection-history {
                margin-top: 0.5rem;
                padding: 0.25rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="detection-history">', unsafe_allow_html=True)
    st.subheader("Detection History")
    history_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process frames if detection is running
    if st.session_state.detection_running and st.session_state.detector is not None:
        fps_list = []  # Store recent FPS values
        
        # Main detection loop
        while st.session_state.detection_running:
            try:
                # Get frame from queue with timeout
                try:
                    frame = frame_queue.get(timeout=1)
                except queue.Empty:
                    st.warning("No frame received from camera. Please check camera connection.")
                    time.sleep(1)
                    continue
                
                # Process frame
                start_time = time.time()
                processed_frame, plate_info = st.session_state.detector.process_frame(
                    frame,
                    use_ocr=True if gemini_key else False,
                    save_plates=st.session_state.save_plates,
                    output_dir=output_dir
                )
                
                # Calculate FPS
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                fps_list.append(fps)
                if len(fps_list) > 10:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                
                # Add FPS and timestamp to the frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    processed_frame,
                    f"FPS: {avg_fps:.1f} | {timestamp}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display processed frame with responsive sizing
                video_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_column_width=True,
                    caption="Live Camera Feed"
                )
                
                # Update status with mobile-friendly layout
                num_plates = len(plate_info)
                status = f"Detected {num_plates} license plate{'s' if num_plates != 1 else ''}"
                if num_plates > 0:
                    status += ": " + ", ".join([
                        f"{info['type']} ({info.get('plate_text', 'N/A')})" 
                        for info in plate_info
                    ])
                status_text.info(status)
                
                # Get detection history and display with mobile-friendly layout
                detection_history = st.session_state.detector.get_detection_history()
                
                if detection_history:
                    # Create DataFrame for display
                    df = pd.DataFrame(detection_history[-10:])  # Show last 10 detections
                    
                    # Sort by timestamp in descending order
                    if 'timestamp' in df.columns:
                        df = df.sort_values('timestamp', ascending=False)
                    
                    # Display with mobile-friendly styling
                    history_placeholder.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                time.sleep(0.01)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                st.error(f"Error in detection loop: {str(e)}")
                time.sleep(1)  # Wait before retrying
                continue
    
    # Display saved detections
    if not st.session_state.detection_running:
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
            st.subheader("Detection Records")
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