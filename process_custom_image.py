import os
import sys
import time
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

# Add parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from license_plate_detector import LicensePlateDetector

def select_image():
    """Open file dialog to select an image file"""
    # Create a tkinter root window
    root = tk.Tk()
    root.title("Select Image")
    
    # Make it appear on top of other windows
    root.attributes('-topmost', True)
    
    # Set window size and position it in the center of the screen
    root.geometry("300x100")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 300) // 2
    y = (screen_height - 100) // 2
    root.geometry(f"300x100+{x}+{y}")
    
    # Add a label with instructions
    label = tk.Label(root, text="Click 'Open File' to select an image")
    label.pack(pady=10)
    
    # Create a variable to store the file path
    file_path = None
    
    # Function to handle file selection
    def open_file_dialog():
        nonlocal file_path
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
    
    # Add a button to open the file dialog
    open_button = tk.Button(root, text="Open File", command=open_file_dialog)
    open_button.pack(pady=10)
    
    # Function to handle window close
    def on_closing():
        nonlocal file_path
        file_path = None
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main loop
    root.mainloop()
    
    return file_path

def select_model():
    """Find available models and let user select one"""
    # First check for the model in the current directory
    local_model_path = os.path.join(script_dir, "best.pt")
    if os.path.exists(local_model_path):
        print(f"Using model from current directory: best.pt")
        return local_model_path
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find trained models
    model_paths = []
    detect_dir = os.path.join(project_root, "runs", "detect")
    if os.path.exists(detect_dir):
        for train_dir in sorted(os.listdir(detect_dir), reverse=True):
            model_path = os.path.join(detect_dir, train_dir, "weights", "best.pt")
            if os.path.exists(model_path):
                model_paths.append((train_dir, model_path))
    
    if not model_paths:
        print("Error: No trained models found.")
        return None
    
    # If only one model, select it automatically
    if len(model_paths) == 1:
        print(f"Using model: {model_paths[0][0]}")
        return model_paths[0][1]
    
    # Otherwise, let user select
    print("Available models:")
    for i, (train_dir, _) in enumerate(model_paths):
        print(f"  {i+1}. {train_dir}")
    
    while True:
        try:
            choice = int(input("\nSelect model (number): "))
            if 1 <= choice <= len(model_paths):
                return model_paths[choice-1][1]
            print(f"Please enter a number between 1 and {len(model_paths)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None

def main():
    """Main function to process a custom image"""
    print("=== License Plate Detector - Custom Image Processing ===")
    
    # Select model
    model_path = select_model()
    if not model_path:
        print("Model selection canceled or no models available.")
        return
    
    # Set Gemini API key directly
    gemini_api_key = "AIzaSyAbaKjFDO5hvcOhb99h_2F9oH5uzrzY6FY"  # Replace with your actual API key
    use_gemini = True
    print("Using Gemini AI for OCR to extract license plate text")
    
    # Initialize detector
    try:
        detector = LicensePlateDetector(
            model_path=model_path,
            device="0" if LicensePlateDetector.is_cuda_available() else "cpu",
            conf_threshold=0.3,
            gemini_api_key=gemini_api_key
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Select and process image
    while True:
        print("\nSelect an image to process (or press Cancel to exit)")
        image_path = select_image()
        
        if not image_path:
            print("Image selection canceled.")
            break
        
        try:
            print(f"Processing image: {image_path}")
            
            # Prepare output path
            output_dir = os.path.join(script_dir, "custom_results")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
            
            # Process the image
            start_time = time.time()
            annotated_img, plate_images, plate_info = detector.process_image(
                image_path,
                save_path=output_path,
                show_result=True,  # Show the result immediately
                use_ocr=use_gemini  # Use OCR if Gemini is enabled
            )
            elapsed_time = time.time() - start_time
            
            # Display results
            print(f"\nDetection completed in {elapsed_time:.3f} seconds")
            print(f"Detected {len(plate_images)} license plates")
            
            for i, info in enumerate(plate_info):
                print(f"  Plate {i+1}: {info['type']} (confidence: {info['confidence']:.2f})")
                if 'plate_text' in info and info['plate_text']:
                    print(f"    Text: {info['plate_text']}")
            
            print(f"\nOutput saved to: {output_path}")
            
            # Ask user if they want to process another image
            if input("\nProcess another image? (y/n): ").lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error processing image: {e}")
            if input("\nTry another image? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main() 