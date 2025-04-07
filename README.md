# License Plate Detector with Gemini OCR

This project detects license plates in images and extracts the text using Gemini AI OCR.

## Features

- Detect license plates in images using YOLO
- Extract text from license plates using Google's Gemini AI Vision model
- Save detection results including plate text to CSV
- Display and save annotated images

## Setup

1. Install the required packages:
   ```
   pip install ultralytics opencv-python google-generativeai pillow
   ```

2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Usage

### Process custom images with OCR

Run the batch file:
```
process_custom_image_with_ocr.bat
```

Or run the Python script directly:
```
python process_custom_image.py
```

When prompted:
1. Select a model to use for detection
2. Choose whether to use Gemini AI for OCR
3. Enter your Gemini API key (or set it as an environment variable named `GEMINI_API_KEY`)
4. Select an image to process
5. View the detected license plates and extracted text

### Results

Detection results are saved in the `custom_results` directory:
- Annotated images with detection boxes and OCR text
- Individual cropped license plate images 
- `plate_detections.csv` with all detected license plates and their extracted text

## Environment Variable

To avoid entering your API key each time, you can set it as an environment variable:

1. Windows: `setx GEMINI_API_KEY "your-api-key-here"`
2. Linux/Mac: `export GEMINI_API_KEY="your-api-key-here"` 