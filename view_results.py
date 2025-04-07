import os
import sys
import cv2
import glob
import argparse
import numpy as np
from tkinter import Tk, Label, Button, Frame, Canvas, NW, filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk

class ResultViewer:
    def __init__(self, root, results_dir):
        self.root = root
        self.root.title("License Plate Detection Results")
        self.root.geometry("1200x800")
        
        # Set the results directory
        self.results_dir = results_dir
        
        # Find all annotated images
        self.image_files = sorted(glob.glob(os.path.join(self.results_dir, "*.jpg")))
        self.current_index = 0
        
        if not self.image_files:
            Label(root, text="No result images found!").pack(pady=20)
            return
        
        # Create the main frame
        self.main_frame = Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create canvas for displaying images
        self.canvas = Canvas(self.main_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # Create controls frame
        self.controls_frame = Frame(root)
        self.controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Create navigation buttons
        self.prev_button = Button(self.controls_frame, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left", padx=5)
        
        self.next_button = Button(self.controls_frame, text="Next", command=self.next_image)
        self.next_button.pack(side="left", padx=5)
        
        # Create a scale for image navigation
        self.scale = Scale(self.controls_frame, from_=1, to=len(self.image_files), 
                          orient=HORIZONTAL, length=400, command=self.scale_changed)
        self.scale.pack(side="left", padx=20)
        
        # Image counter label
        self.counter_label = Label(self.controls_frame, text=f"Image 1/{len(self.image_files)}")
        self.counter_label.pack(side="left", padx=20)
        
        # Bind arrow keys for navigation
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        
        # Display first image
        self.display_image(0)
    
    def display_image(self, index):
        if 0 <= index < len(self.image_files):
            # Read the image
            self.current_index = index
            img_path = self.image_files[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = img.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
            
            # Update counter and scale
            self.counter_label.config(text=f"Image {index+1}/{len(self.image_files)}")
            self.scale.set(index+1)
            
            # Extract filename
            filename = os.path.basename(img_path)
            self.root.title(f"License Plate Detection - {filename}")
    
    def prev_image(self):
        self.display_image((self.current_index - 1) % len(self.image_files))
    
    def next_image(self):
        self.display_image((self.current_index + 1) % len(self.image_files))
    
    def scale_changed(self, value):
        self.display_image(int(float(value)) - 1)

def main():
    parser = argparse.ArgumentParser(description='View License Plate Detection Results')
    parser.add_argument('--dir', type=str, default='results', 
                        help='Directory containing result images')
    args = parser.parse_args()
    
    # Find the results directory
    if not os.path.isabs(args.dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, args.dir)
    else:
        results_dir = args.dir
    
    # Check if directory exists
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the detection script first.")
        return
    
    # Initialize the Tkinter root
    root = Tk()
    app = ResultViewer(root, results_dir)
    root.mainloop()

if __name__ == "__main__":
    main() 