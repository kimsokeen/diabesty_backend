from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response # Although Response is imported, it's not used in this endpoint, but keeping it as per original
import tensorflow as tf
from tensorflow.keras import backend as K
# No longer need to import load_model directly here, as we'll use tf.keras.models.load_model
from PIL import Image
from PIL import ImageOps
import numpy as np
import io
import base64
import cv2
import os # Import os for environment variables and path handling

# --- 1. Global variables to hold the loaded models ---
# Initialize them to None. They will be populated during the startup event.
classifier_model = None
segmentation_model = None

# Custom functions for segmentation model
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + 1e-6)

# FastAPI setup
app = FastAPI()

# --- 2. Define the startup event handler for loading models ---
@app.on_event("startup")
async def load_models_on_startup():
    """
    Loads the Keras classification and segmentation models into memory
    when the FastAPI application starts. This ensures models are loaded
    only once, not for every request.
    """
    global classifier_model
    global segmentation_model

    # Define model paths using environment variables for flexibility
    # Default paths are provided for local testing if env vars are not set
    classifier_model_path = os.getenv("CLASSIFIER_MODEL_PATH", "models/diabetic_foot_ulcer_classifier_final.keras")
    segmentation_model_path = os.getenv("SEGMENTATION_MODEL_PATH", "models/foot_ulcer_model_mobilenet.keras")

    print(f"Attempting to load classifier model from: {classifier_model_path}")
    try:
        classifier_model = tf.keras.models.load_model(classifier_model_path)
        classifier_model.summary() # Print summary to confirm loading
        print("Classifier model loaded successfully during startup.")
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        raise RuntimeError(f"Failed to load classifier ML model: {e}")

    print(f"Attempting to load segmentation model from: {segmentation_model_path}")
    try:
        segmentation_model = tf.keras.models.load_model(
            segmentation_model_path,
            custom_objects={"dice_loss": dice_loss, "iou_metric": iou_metric}
        )
        segmentation_model.summary() # Print summary to confirm loading
        print("Segmentation model loaded successfully during startup.")
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        raise RuntimeError(f"Failed to load segmentation ML model: {e}")


# --- CORS Middleware (as per your original code) ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://diabest.netlify.app" # <--- ADD YOUR NETLIFY FRONTEND URL HERE
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# --- Helper functions (unchanged) ---
def analyze_hsv_from_mask(original_img, mask):
    binary_mask = (mask > 0).astype(np.uint8)
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)

    wound_pixels = hsv_img[binary_mask == 1]

    if wound_pixels.size == 0:
        return {
            "red_area_percent": 0,
            "yellow_area_percent": 0,
            "black_area_percent": 0,
        }

    wound_pixels_reshaped = wound_pixels.reshape(-1, 1, 3)

    red_lower_1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper_1 = np.array([10, 255, 255], dtype=np.uint8)
    
    red_lower_2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper_2 = np.array([179, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([20, 20, 100], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([179, 50, 50], dtype=np.uint8)

    result = {}
    total_wound_pixels = wound_pixels_reshaped.shape[0]

    mask_red_1 = cv2.inRange(wound_pixels_reshaped, red_lower_1, red_upper_1)
    mask_red_2 = cv2.inRange(wound_pixels_reshaped, red_lower_2, red_upper_2)
    mask_red_combined = cv2.bitwise_or(mask_red_1, mask_red_2)
    count_red = np.count_nonzero(mask_red_combined)
    result["red_area_percent"] = round((count_red / total_wound_pixels) * 100, 2)
    print(f"Red pixels: {count_red} ({result['red_area_percent']}%) out of {total_wound_pixels}")

    mask_yellow = cv2.inRange(wound_pixels_reshaped, yellow_lower, yellow_upper)
    count_yellow = np.count_nonzero(mask_yellow)
    result["yellow_area_percent"] = round((count_yellow / total_wound_pixels) * 100, 2)
    print(f"Yellow pixels: {count_yellow} ({result['yellow_area_percent']}%) out of {total_wound_pixels}")

    mask_black = cv2.inRange(wound_pixels_reshaped, black_lower, black_upper)
    count_black = np.count_nonzero(mask_black)
    result["black_area_percent"] = round((count_black / total_wound_pixels) * 100, 2)
    print(f"Black pixels: {count_black} ({result['black_area_percent']}%) out of {total_wound_pixels}")

    return result

def detect_reference_coin_radius(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=60
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = image.shape[:2]
        bottom_right_circle = max(
            circles[0], key=lambda c: (c[0] + c[1])
        )
        return bottom_right_circle
    return None

def pad_to_square(image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    width, height = image.size
    max_side = max(width, height)
    delta_w = max_side - width
    delta_h = max_side - height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=fill_color)

# --- 3. Your /upload/ endpoint (now using global models) ---
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    """
    Handles image upload, processes it using the loaded ML models,
    and returns prediction results.
    """
    # Safeguard: Ensure models are loaded before processing requests
    if classifier_model is None or segmentation_model is None:
        raise HTTPException(status_code=503, detail="ML models are not loaded yet. Please try again in a moment.")

    contents = await file.read()

    # Classification Preprocessing
    original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    padded_image = pad_to_square(original_image)

    # Resize only AFTER padding
    image = padded_image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_input = np.expand_dims(image_array, axis=0)

    # Classify using the global classifier_model
    prediction = classifier_model.predict(image_input)[0][0]
    predicted_class = "non-diabetic foot" if prediction >= 0.7 else "diabetic foot"

    response_data = {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": float(prediction)
    }

    # If diabetic, run segmentation
    if prediction < 0.7:
        seg_input = np.expand_dims(np.array(padded_image.resize((256, 256))) / 255.0, axis=0)
        # Predict using the global segmentation_model
        mask = segmentation_model.predict(seg_input)[0]

        # Convert to binary mask
        binary_mask = (mask > 0.3).astype(np.uint8)

        # Count pixels
        wound_area = int(np.sum(binary_mask))

        # Convert mask to image
        mask_image = (binary_mask.squeeze() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image)
        buffered = io.BytesIO()
        mask_pil.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Resize original image for HSV and coin detection
        original_for_hsv = np.array(padded_image.resize((256, 256)))

        hsv_stats = analyze_hsv_from_mask(original_for_hsv, mask_image)
        response_data["hsv_stats"] = hsv_stats

        circle = detect_reference_coin_radius(original_for_hsv)
        if circle is not None:
            x, y, radius = circle
            coin_area = np.pi * (radius ** 2)

            real_wound_cm2 = (wound_area / coin_area) * 2.0  # assuming coin = 2cm²
            response_data["coin_radius_px"] = int(radius)
            response_data["coin_area_px"] = int(coin_area)
            response_data["wound_area_cm2"] = round(real_wound_cm2, 2)

            # draw detected coin
            circle_img = original_for_hsv.copy()
            cv2.circle(circle_img, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(circle_img, (x, y), 2, (255, 0, 0), 3)

            # Encode image with circle
            _, buffer = cv2.imencode('.png', circle_img)
            circle_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["circle_image_base64"] = circle_base64
        else:
            response_data["coin_radius_px"] = None
            response_data["wound_area_cm2"] = None
            response_data["circle_image_base64"] = None

        response_data["wound_area_pixels"] = int(wound_area)
        response_data["mask_base64"] = mask_base64
        response_data["hsv_stats"] = {k: float(v) for k, v in hsv_stats.items()}

    return JSONResponse(content=response_data)
