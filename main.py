from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
from PIL import ImageOps
import numpy as np
import io
import base64
import cv2


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

# Load models
model = tf.keras.models.load_model("models/mobilenet_model1.h5")
seg_model = load_model(
    "models/foot_ulcer_model_mobilenet.keras",
    custom_objects={"dice_loss": dice_loss, "iou_metric": iou_metric}
)

# FastAPI setup
app = FastAPI()

# --- START OF CHANGE ---
# IMPORTANT: Replace "https://YOUR_NETLIFY_APP_URL.netlify.app" with your actual Netlify URL
# You can find your Netlify URL in your Netlify dashboard after your frontend is deployed.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://diabest.netlify.app" # <--- ADD YOUR NETLIFY FRONTEND URL HERE
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
    allow_credentials=True # Important if you use cookies/authentication
)
# --- END OF CHANGE ---

def analyze_hsv_from_mask(original_img, mask):
    binary_mask = (mask > 0).astype(np.uint8)
    # Ensure original_img is in BGR if it's coming from typical OpenCV loading,
    # as cv2.COLOR_RGB2HSV expects RGB, but default imread is BGR.
    # If your original_img is already guaranteed to be RGB, then this check is not strictly needed.
    # For safety, let's assume original_img might be BGR if coming from typical CV2 operations
    # and convert it to RGB first if it's not already. Or, more simply, use BGR2HSV directly.
    # Let's assume original_img is in RGB format as per your original code's cv2.COLOR_RGB2HSV.
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)

    # Mask the wound area
    wound_pixels = hsv_img[binary_mask == 1]

    if wound_pixels.size == 0:
        return {
            "red_area_percent": 0,
            "yellow_area_percent": 0,
            "black_area_percent": 0,
        }

    # Reshape to (N, 1, 3) so cv2.inRange works correctly for individual pixels
    # For a list of pixels, you can also use wound_pixels directly with inRange if it's (N, 3)
    # but reshaping to (N, 1, 3) is a safe explicit way for cv2.inRange
    wound_pixels_reshaped = wound_pixels.reshape(-1, 1, 3)

    # --- UPDATED COLOR RANGES ---
    # Using the suggested HSV ranges (OpenCV's H: 0-179, S: 0-255, V: 0-255)
    # Red needs two ranges due to wrapping
    red_lower_1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper_1 = np.array([10, 255, 255], dtype=np.uint8)
    
    red_lower_2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper_2 = np.array([179, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([20, 20, 100], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([179, 50, 50], dtype=np.uint8) # Hue doesn't matter for black, Saturation and Value are low

    # --- Processing Colors ---
    result = {}
    total_wound_pixels = wound_pixels_reshaped.shape[0]

    # Calculate Red area
    mask_red_1 = cv2.inRange(wound_pixels_reshaped, red_lower_1, red_upper_1)
    mask_red_2 = cv2.inRange(wound_pixels_reshaped, red_lower_2, red_upper_2)
    mask_red_combined = cv2.bitwise_or(mask_red_1, mask_red_2)
    count_red = np.count_nonzero(mask_red_combined)
    result["red_area_percent"] = round((count_red / total_wound_pixels) * 100, 2)
    print(f"Red pixels: {count_red} ({result['red_area_percent']}%) out of {total_wound_pixels}")


    # Calculate Yellow area
    mask_yellow = cv2.inRange(wound_pixels_reshaped, yellow_lower, yellow_upper)
    count_yellow = np.count_nonzero(mask_yellow)
    result["yellow_area_percent"] = round((count_yellow / total_wound_pixels) * 100, 2)
    print(f"Yellow pixels: {count_yellow} ({result['yellow_area_percent']}%) out of {total_wound_pixels}")


    # Calculate Black area
    mask_black = cv2.inRange(wound_pixels_reshaped, black_lower, black_upper)
    count_black = np.count_nonzero(mask_black)
    result["black_area_percent"] = round((count_black / total_wound_pixels) * 100, 2)
    print(f"Black pixels: {count_black} ({result['black_area_percent']}%) out of {total_wound_pixels}")

    return result

def detect_reference_coin_radius(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # Hough Circle Transform
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
        # Choose the one in bottom-right corner
        height, width = image.shape[:2]
        bottom_right_circle = max(
            circles[0], key=lambda c: (c[0] + c[1])  # bottom-right = high x+y
        )
        return bottom_right_circle  # x, y, radius
    return None

def pad_to_square(image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    """
    Pads the image to a square using black (or specified color).
    """
    width, height = image.size
    max_side = max(width, height)
    delta_w = max_side - width
    delta_h = max_side - height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=fill_color)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()

    # Classification Preprocessing
    original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    padded_image = pad_to_square(original_image)

    # Resize only AFTER padding
    image = padded_image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_input = np.expand_dims(image_array, axis=0)

    # Classify
    prediction = model.predict(image_input)[0][0]
    predicted_class = "non-diabetic foot" if prediction >= 0.5 else "diabetic foot"

    response_data = {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": float(prediction)
    }

    # If diabetic, run segmentation
    if prediction < 0.5:
        seg_input = np.expand_dims(np.array(padded_image.resize((256, 256))) / 255.0, axis=0)
        mask = seg_model.predict(seg_input)[0]

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

                # Resize original image
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