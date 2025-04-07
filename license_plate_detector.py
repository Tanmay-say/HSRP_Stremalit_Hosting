import cv2
import numpy as np
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import argparse
import time
import google.generativeai as genai
from PIL import Image
import io

class LicensePlateDetector:
    def __init__(self, model_path, device='0', conf_threshold=0.5, gemini_api_key=None):
        """
        Initialize the License Plate Detector
        
        Args:
            model_path: Path to the trained YOLOv8 model
            device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
            conf_threshold: Confidence threshold for detections
            gemini_api_key: API key for Google's Gemini AI model for OCR
        """
        # Check if CUDA is available when device is set to GPU
        if device != 'cpu' and not torch.cuda.is_available():
            print("CUDA is not available, defaulting to CPU")
            device = 'cpu'
            
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load the YOLO model
        print(f"Loading model from {model_path} on device {device}")
        self.model = YOLO(model_path)
        
        # Class names
        self.class_names = ['ordinary', 'hsrp']
        self.class_colors = [(0, 255, 0), (0, 0, 255)]  # Green for ordinary, Blue for HSRP
        
        # Initialize Gemini for OCR if API key is provided
        self.gemini_available = False
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                # Update to use Gemini 1.5 Flash instead of pro-vision (which is deprecated)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
                print("Gemini OCR initialized successfully")
            except Exception as e:
                print(f"Error initializing Gemini OCR: {e}")
        
    @staticmethod
    def is_cuda_available():
        """Check if CUDA is available for GPU acceleration"""
        return torch.cuda.is_available()
        
    def detect(self, image):
        """
        Detect license plates in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            results: Model detection results
            annotated_img: Image with detection annotations
        """
        # Perform detection
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        return results[0], annotated_img
    
    def extract_license_plates(self, image, results):
        """
        Extract license plate regions from detection results
        
        Args:
            image: Original image
            results: Detection results from YOLO model
            
        Returns:
            plate_images: List of cropped license plate images
            plate_info: List of dictionaries with plate information
        """
        plate_images = []
        plate_info = []
        
        height, width = image.shape[:2]
        
        if len(results.boxes) == 0:
            return [], []
            
        for i, box in enumerate(results.boxes):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Get class and confidence
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            
            # Crop the license plate
            plate_img = image[y1:y2, x1:x2].copy()
            
            # Add to lists
            plate_images.append(plate_img)
            plate_info.append({
                'type': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'plate_text': None  # Will be filled by OCR if available
            })
            
        return plate_images, plate_info

    def extract_text_with_gemini(self, plate_image):
        """
        Extract text from license plate image using Gemini Vision API
        
        Args:
            plate_image: Cropped license plate image (numpy array)
            
        Returns:
            plate_text: Extracted text from license plate
        """
        if not self.gemini_available:
            return None
            
        try:
            # Convert OpenCV image (numpy array) to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            
            # Create byte stream for image
            byte_stream = io.BytesIO()
            pil_image.save(byte_stream, format='JPEG')
            byte_stream.seek(0)
            
            # Prepare prompt for Gemini
            prompt = """
            Extract the license plate number from this image. 
            Return ONLY the text content of the license plate without any additional text or explanations.
            If no text is clearly visible, respond with 'UNREADABLE'.
            """
            
            # Generate response from Gemini Vision
            response = self.gemini_model.generate_content([prompt, {"mime_type": "image/jpeg", "data": byte_stream.getvalue()}])
            
            # Get plate text
            plate_text = response.text.strip()
            return plate_text
            
        except Exception as e:
            print(f"Error in Gemini OCR: {e}")
            return None
    
    def process_image(self, image, show_result=True, save_path=None, use_ocr=True):
        """
        Process an image to detect and extract license plates
        
        Args:
            image: Input image (file path or numpy array)
            show_result: Whether to attempt displaying the result (may not work in all environments)
            save_path: Path to save the result (None to not save)
            use_ocr: Whether to use OCR to extract text from license plates
            
        Returns:
            annotated_img: Image with detection annotations
            plate_images: List of cropped license plate images
            plate_info: List of dictionaries with plate information
        """
        # Load image if path is provided
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
            if image is None:
                print(f"Error: Could not load image {image}")
                return None, [], []
                
        # Detect license plates
        results, annotated_img = self.detect(image)
        
        # Extract license plate regions
        plate_images, plate_info = self.extract_license_plates(image, results)
        
        # Extract text using Gemini OCR if available and enabled
        if use_ocr and self.gemini_available:
            for i, plate_img in enumerate(plate_images):
                if plate_img.size > 0:
                    print(f"Extracting text from plate {i+1}...")
                    plate_text = self.extract_text_with_gemini(plate_img)
                    plate_info[i]['plate_text'] = plate_text
                    
                    # Add text to the annotated image
                    if plate_text:
                        x1, y1, x2, y2 = plate_info[i]['bbox']
                        cv2.putText(
                            annotated_img,
                            f"Text: {plate_text}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
        
        # Skip displaying with OpenCV as it might not be supported in all environments
        # Instead just print the detection info
        if show_result:
            print("\nDetection Results:")
            print(f"- Found {len(plate_images)} license plates")
            for i, info in enumerate(plate_info):
                plate_title = f"Plate {i+1}: {info['type']} (confidence: {info['confidence']:.2f})"
                if info.get('plate_text'):
                    plate_title += f" - Text: {info['plate_text']}"
                print(f"- {plate_title}")
            
        # Save results if path is provided
        if save_path is not None:
            cv2.imwrite(save_path, annotated_img)
            
            # Save cropped plates
            save_dir = Path(save_path).parent / "plates"
            save_dir.mkdir(exist_ok=True)
            
            # Create a CSV file to store plate data
            detection_data_file = Path(save_path).parent / "plate_detections.csv"
            is_new_file = not detection_data_file.exists()
            
            with open(detection_data_file, 'a') as f:
                # Write header if it's a new file
                if is_new_file:
                    f.write("image_name,plate_number,plate_type,confidence,plate_text\n")
                
                # Write plate info
                image_name = Path(save_path).name
                for i, (plate_img, info) in enumerate(zip(plate_images, plate_info)):
                    if plate_img.size > 0:
                        plate_type = info['type']
                        conf = info['confidence']
                        plate_text = info.get('plate_text', 'N/A')
                        
                        # Write to CSV
                        f.write(f"{image_name},{i+1},{plate_type},{conf:.2f},{plate_text}\n")
                        
                        # Save plate image
                        plate_filename = Path(save_path).stem
                        plate_save_path = save_dir / f"{plate_filename}_plate{i+1}_{plate_type}_{conf:.2f}.jpg"
                        cv2.imwrite(str(plate_save_path), plate_img)
            
            print(f"\nResults saved to {save_path}")
            print(f"Plate images saved to {save_dir}")
            print(f"Detection data saved to {detection_data_file}")
            
        return annotated_img, plate_images, plate_info
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video to detect license plates
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (None to not save)
            display: Whether to display the video while processing
            
        Returns:
            all_plate_info: List of dictionaries with plate information for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        if output_path is not None:
            output_dir = Path(output_path).parent
            output_dir.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_plate_info = []
        frame_idx = 0
        
        print(f"Processing video with {frame_count} frames...")
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results, annotated_frame = self.detect(frame)
            
            # Extract plate information
            plate_images, plate_info = self.extract_license_plates(frame, results)
            all_plate_info.append(plate_info)
            
            # Add frame number and FPS info
            cv2.putText(
                annotated_frame, 
                f"Frame: {frame_idx} / {frame_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            elapsed_time = time.time() - start_time
            processed_fps = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(
                annotated_frame, 
                f"Processing FPS: {processed_fps:.1f}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Write frame if output path is provided
            if output_path is not None:
                out.write(annotated_frame)
            
            # Display frame
            if display:
                cv2.imshow("License Plate Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            
            # Print progress every 10 frames
            if frame_idx % 10 == 0:
                progress = frame_idx / frame_count * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count})")
        
        cap.release()
        if output_path is not None:
            out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({frame_count/total_time:.2f} FPS)")
        
        return all_plate_info
    
    def process_directory(self, input_dir, output_dir=None, show_result=False):
        """
        Process all images in a directory
        
        Args:
            input_dir: Path to directory containing images
            output_dir: Path to save output images (None to not save)
            show_result: Whether to display results
            
        Returns:
            results_dict: Dictionary with results for each image
        """
        input_dir = Path(input_dir)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
            image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return {}
            
        print(f"Processing {len(image_files)} images...")
        
        results_dict = {}
        for i, img_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
            
            # Determine output path
            save_path = None
            if output_dir is not None:
                save_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
            
            # Process image
            annotated_img, plate_images, plate_info = self.process_image(
                str(img_path), 
                show_result=show_result,
                save_path=save_path
            )
            
            # Store results
            results_dict[img_path.name] = {
                'plate_info': plate_info
            }
        
        return results_dict

def main():
    parser = argparse.ArgumentParser(description="License Plate Detection System")
    parser.add_argument('--model', type=str, default=None, help='Path to YOLOv8 model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image, video, or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to save output')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--show', action='store_true', help='Display results')
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model is None:
        # Look for trained model
        model_paths = list(Path('runs/detect').glob('*/weights/best.pt'))
        if not model_paths:
            print("No trained model found. Please specify a model path with --model")
            return
        args.model = str(model_paths[0])
        print(f"Using model: {args.model}")
    
    # Initialize detector
    detector = LicensePlateDetector(args.model, args.device, args.conf)
    
    input_path = Path(args.input)
    
    # Process based on input type
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process video
            detector.process_video(str(input_path), args.output, args.show)
        else:
            # Process single image
            detector.process_image(str(input_path), args.show, args.output)
    elif input_path.is_dir():
        # Process directory of images
        detector.process_directory(str(input_path), args.output, args.show)
    else:
        print(f"Error: Input path {args.input} does not exist")

if __name__ == "__main__":
    main() 