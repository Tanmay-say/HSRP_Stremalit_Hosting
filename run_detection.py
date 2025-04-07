import os
import sys
import cv2
import time
import tqdm
from pathlib import Path

# Add parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(script_dir)

from license_plate_detector import LicensePlateDetector

def main():
    """
    Main function to run license plate detection on test images
    """
    print("=== License Plate Detection System ===")
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find the latest model
    model_paths = []
    for train_dir in sorted(os.listdir(os.path.join(project_root, "runs", "detect")), reverse=True):
        model_path = os.path.join(project_root, "runs", "detect", train_dir, "weights", "best.pt")
        if os.path.exists(model_path):
            model_paths.append((train_dir, model_path))
    
    if not model_paths:
        print("Error: No trained models found. Please train a model first.")
        return
    
    # Let user select model if multiple are available
    selected_model_path = None
    if len(model_paths) == 1:
        selected_model_path = model_paths[0][1]
    else:
        print("\nMultiple trained models found:")
        for i, (train_dir, path) in enumerate(model_paths):
            print(f"  {i+1}. {train_dir}")
        
        while True:
            try:
                choice = int(input("\nSelect model to use (number): "))
                if 1 <= choice <= len(model_paths):
                    selected_model_path = model_paths[choice-1][1]
                    break
                print(f"Please enter a number between 1 and {len(model_paths)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Setup test directory
    test_images_dir = os.path.join(project_root, "data", "test", "images")
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        return
    
    # Setup output directory
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model: {selected_model_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize license plate detector
    detector = LicensePlateDetector(
        model_path=selected_model_path,
        device="0" if LicensePlateDetector.is_cuda_available() else "cpu",
        conf_threshold=0.3
    )
    
    print("\nProcessing test images...")
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(test_images)} images...")
    
    # Statistics for tracking results
    total_plates = 0
    ordinary_plates = 0
    hsrp_plates = 0
    processing_times = []
    
    # Process all test images with progress bar
    for img_file in tqdm.tqdm(test_images, desc="Processing images"):
        img_path = os.path.join(test_images_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        start_time = time.time()
        _, plate_images, plate_info = detector.process_image(
            img_path, 
            save_path=output_path,
            show_result=False
        )
        end_time = time.time()
        
        # Update statistics
        processing_times.append(end_time - start_time)
        total_plates += len(plate_images)
        
        # Count plate types
        for info in plate_info:
            if info["type"].lower() == "ordinary":
                ordinary_plates += 1
            elif info["type"].lower() == "hsrp":
                hsrp_plates += 1
    
    # Calculate and display statistics
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    print("\n=== Detection Results ===")
    print(f"Total images processed: {len(test_images)}")
    print(f"Total plates detected: {total_plates}")
    print(f"Ordinary plates: {ordinary_plates}")
    print(f"HSRP plates: {hsrp_plates}")
    print(f"Average processing time: {avg_time:.4f} seconds per image")
    print(f"FPS: {1/avg_time:.2f}" if avg_time > 0 else "FPS: N/A")
    print("\nResults saved to:", output_dir)
    print("\nRun 'view_results.bat' or 'python view_results.py' to view the results")

if __name__ == "__main__":
    main() 