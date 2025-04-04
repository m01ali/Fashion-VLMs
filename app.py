from flask import Flask, request, send_file, jsonify
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import io
import uuid
import traceback

# Try to import pillow_avif to add AVIF support
try:
    import pillow_avif
except ImportError:
    print("WARNING: pillow_avif is not installed. AVIF support will be limited.")
    print("To add AVIF support, install: pip install pillow-avif-plugin")

# Add the U-2-Net repository to the Python path
# Assuming U-2-Net is cloned in the same directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
u2net_dir = os.path.join(current_dir, 'U-2-Net')
sys.path.append(u2net_dir)

# Import the U2NET model
try:
    from model import U2NET
except ImportError:
    print("ERROR: Could not import U2NET model. Please make sure you've cloned the U-2-Net repository.")
    print("Run: git clone https://github.com/xuebinqin/U-2-Net.git")
    sys.exit(1)

app = Flask(__name__)

# Constants
MODEL_DIR = os.path.join(current_dir, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'u2net.pth')
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
OUTPUT_FOLDER = os.path.join(current_dir, 'outputs')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper function to download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"Model file not found at {MODEL_PATH}. Attempting to download...")
            
            try:
                from huggingface_hub import hf_hub_download
                
                # Download the model from Hugging Face
                model_file = hf_hub_download(repo_id="lilpotat/pytorch3d", filename="u2net.pth", cache_dir=MODEL_DIR)
                
                # If the download puts it in a different location, copy or move it to our expected path
                if model_file != MODEL_PATH:
                    import shutil
                    shutil.copy(model_file, MODEL_PATH)
                
                print(f"Model downloaded successfully to {MODEL_PATH}")
                
            except ImportError:
                print("huggingface_hub not installed. Please install it using: pip install huggingface_hub")
                print("Then manually download the model or rerun the application.")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            print("Please download the model manually and place it at:", MODEL_PATH)
            sys.exit(1)

# Helper function to normalize the prediction map to [0, 1]
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

# Helper function to handle various image formats including AVIF
def open_image_safe(image_path):
    try:
        # Try to open the image normally first
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        # If normal opening fails, try using a fallback method for AVIF
        if image_path.lower().endswith('.avif'):
            try:
                # If pillow_avif didn't import, try one more fallback approach
                # Convert AVIF to another format using external tool if available
                print(f"Attempting conversion for AVIF file: {image_path}")
                
                # Try using OpenCV as fallback
                img_cv = cv2.imread(image_path)
                if img_cv is not None:
                    # Convert from BGR to RGB
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    image = Image.fromarray(img_cv)
                    return image
                else:
                    raise Exception("Failed to read AVIF file with OpenCV")
            except Exception as avif_error:
                print(f"AVIF conversion failed: {str(avif_error)}")
                raise Exception(f"Failed to process AVIF image: {str(e)}. Make sure 'pillow-avif-plugin' is installed.")
        # Re-raise original exception for other formats
        raise

# Inference function: preprocess the image, run the model, and get the saliency map
def inference(model, image_path):
    # Open image with enhanced handler for AVIF
    print(f"Opening image from: {image_path}")
    try:
        image = open_image_safe(image_path)
    except Exception as e:
        print(f"Error opening image: {str(e)}")
        raise
    
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = Variable(image_tensor)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        model = model.cuda()
    
    # U2NET returns 7 outputs; we use d1 as the primary saliency map
    print("Running model inference...")
    with torch.no_grad():  # Add this to avoid unnecessary gradient calculation
        d1, d2, d3, d4, d5, d6, d7 = model(image_tensor)
    
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    pred = pred.squeeze().cpu().data.numpy()
    
    return pred

# Process the prediction and save output
def save_output(image_path, pred, output_path):
    # Re-open the image to get the original dimensions
    print(f"Processing prediction and saving to: {output_path}")
    try:
        image = open_image_safe(image_path)
        image = np.array(image)
    except Exception as e:
        print(f"Error reopening image: {str(e)}")
        # Use a default size if we can't open the original
        image = np.zeros((320, 320, 3), dtype=np.uint8)
    
    # Process prediction
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    ret, mask = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
    
    # Save the result
    cv2.imwrite(output_path, mask)
    
    return output_path

# Load the U2NET model
def load_model():
    print("Loading U-2-Net model...")
    
    # Ensure model exists
    download_model()
    
    # Initialize the model
    model = U2NET(3, 1)
    
    # Load the state dictionary
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded on GPU")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("Model loaded on CPU")
    
    model.eval()
    print("Model loaded successfully!")
    return model

# Initialize the model
print(f"Looking for model at: {MODEL_PATH}")
model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "U-2-Net"}), 200

@app.route('/extract-silhouette', methods=['POST'])
def extract_silhouette():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "Empty file provided"}), 400
    
    # Define paths before the try block to ensure they're available in finally
    input_path = None
    output_path = None
    
    try:
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext:
            ext = '.jpg'  # Default extension if none is provided
            
        input_filename = f"{unique_id}_input{ext}"
        output_filename = f"{unique_id}_silhouette.png"
        
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save the uploaded file
        print(f"Saving uploaded file to: {input_path}")
        file.save(input_path)
        
        # Verify the file exists and is readable
        if not os.path.exists(input_path):
            return jsonify({"error": "Failed to save uploaded file"}), 500
            
        # Check if file is actually an image
        try:
            if input_path.lower().endswith('.avif'):
                print("Detected AVIF file format, using special handling")
                # Special handling for AVIF files
                img = open_image_safe(input_path)
                img_format = "AVIF"  # Since we know it's AVIF
                img_size = img.size
            else:
                # Normal handling for other formats
                with Image.open(input_path) as img:
                    img_format = img.format
                    img_size = img.size
            print(f"Image format: {img_format}, size: {img_size}")
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
        # Process the image
        pred = inference(model, input_path)
        save_output(input_path, pred, output_path)
        
        # Verify the output file exists
        if not os.path.exists(output_path):
            return jsonify({"error": "Failed to generate output image"}), 500
            
        # Return the processed image
        print(f"Returning processed image: {output_path}")
        return send_file(output_path, mimetype='image/png')
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error processing image: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({"error": str(e), "traceback": error_details}), 500
    
    finally:
        # Clean up temporary files (optional)
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                # In production, you might want to keep outputs for a while
                os.remove(output_path)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    # Serve a simple HTML page for testing
    html_path = os.path.join(current_dir, 'test_form.html')
    
    # Check if the test form exists, if not create it
    if not os.path.exists(html_path):
        with open(html_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U-2-Net Silhouette Extractor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .result img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        .image-container {
            flex: 1;
            min-width: 300px;
        }
        .error-message {
            color: red;
            font-weight: bold;
            margin-top: 10px;
            padding: 10px;
            border: 1px solid red;
            background-color: #ffeeee;
            display: none;
        }
    </style>
</head>
<body>
    <h1>U-2-Net Silhouette Extractor</h1>
    <div class="upload-form">
        <h2>Upload an Image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageFile" name="image" accept="image/*" required>
            <button type="submit">Extract Silhouette</button>
        </form>
        <div id="errorMessage" class="error-message"></div>
    </div>
    
    <div class="result" id="result" style="display: none;">
        <div class="image-container">
            <h3>Original Image</h3>
            <img id="originalImage" src="" alt="Original Image">
        </div>
        <div class="image-container">
            <h3>Silhouette</h3>
            <img id="silhouetteImage" src="" alt="Extracted Silhouette">
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('imageFile');
            const errorElement = document.getElementById('errorMessage');
            
            // Hide previous error messages
            errorElement.style.display = 'none';
            errorElement.textContent = '';
            
            if (fileInput.files.length === 0) {
                errorElement.textContent = 'Please select an image file';
                errorElement.style.display = 'block';
                return;
            }
            
            formData.append('image', fileInput.files[0]);
            
            try {
                // Display the original image
                const originalImage = document.getElementById('originalImage');
                originalImage.src = URL.createObjectURL(fileInput.files[0]);
                
                // Send the request to the API
                const response = await fetch('/extract-silhouette', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'API request failed');
                }
                
                // Get the response as blob
                const blob = await response.blob();
                
                // Display the silhouette image
                const silhouetteImage = document.getElementById('silhouetteImage');
                silhouetteImage.src = URL.createObjectURL(blob);
                
                // Show the result section
                document.getElementById('result').style.display = 'flex';
                
            } catch (error) {
                console.error('Error:', error);
                errorElement.textContent = error.message || 'Error processing the image. Please try again.';
                errorElement.style.display = 'block';
            }
        });
    </script>
</body>
</html>
            ''')
    
    return send_file(html_path)

if __name__ == '__main__':
    print("Starting U-2-Net Silhouette Extraction API...")
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode for more detailed errors