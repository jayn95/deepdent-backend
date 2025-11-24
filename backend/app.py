from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"


# --- Modular Hugging Face Calls ---

def call_gingivitis_model(image_path, timeout_seconds=240):
    """
    Calls the Hugging Face gingivitis Space.
    Ensures images are returned as Base64 strings for client display.
    """
    client = Client(GINGIVITIS_SPACE)
    result_container = {}

    def run_predict():
        # Send file to HF Space predict endpoint
        result_container["data"] = client.predict(
            handle_file(image_path),
            0.4,
            0.5,
            api_name="/predict"
        )

    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Gingivitis model timed out after {timeout_seconds}s")

    result = result_container.get("data", [])

    # Ensure we have 4 items: swelling, redness, bleeding, diagnosis
    if len(result) != 4:
        raise ValueError(f"Unexpected result from gingivitis model: {result}")

    labels = ["swelling", "redness", "bleeding"]
    images = {}

    for label, item in zip(labels, result[:3]):
        # If item is already a Base64 string, use it
        if isinstance(item, str):
            if os.path.exists(item):
                # Convert server path to Base64
                with open(item, "rb") as f:
                    images[label] = base64.b64encode(f.read()).decode("utf-8")
            else:
                # Assume it is already Base64
                images[label] = item
        else:
            # If item is a PIL/numpy image, convert to Base64
            try:
                import io
                from PIL import Image
                if hasattr(item, "save"):
                    buf = io.BytesIO()
                    item.save(buf, format="JPEG")
                    images[label] = base64.b64encode(buf.getvalue()).decode("utf-8")
                else:
                    # For numpy arrays
                    from PIL import Image
                    import numpy as np
                    pil_img = Image.fromarray(item)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    images[label] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error converting {label} to Base64: {e}")
                images[label] = None

    diagnosis = result[3]  # The diagnosis text

    return {"images": images, "diagnosis": diagnosis}


def call_periodontitis_model(image_path, timeout_seconds=240):
    """
    Calls the periodontitis HF Space which returns:
    1) Numpy image (combined_rgb)
    2) summary_text
    """
    client = Client(PERIODONTITIS_SPACE)
    result_container = {}

    def run_predict():
        result_container["data"] = client.predict(
            handle_file(image_path),
            api_name="/predict"
        )

    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError("Periodontitis model timed out")

    data = result_container.get("data")
    if not data:
        raise ValueError("HF Space returned empty result")

    # Unpack HF returned data
    combined_img = data[0]  # numpy array (RGB)
    summary_text = data[1]  # string

    # Convert image → Base64 so Flutter can display it
    try:
        import cv2, base64
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        img_b64 = base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")

    return {
        "annotated_image": img_b64,
        "analys

# --- Routes ---

@app.route("/")
def home():
    return jsonify({"status": "DeepDent backend running successfully!"})


@app.route("/predict/gingivitis", methods=["POST"])
def predict_gingivitis():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        response = call_gingivitis_model(temp_path)
        return jsonify({
            "images": response["images"],       # Base64 images
            "diagnosis": response.get("diagnosis") or "No diagnosis returned"
        })
    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)


@app.route("/predict/periodontitis", methods=["POST"])
def predict_periodontitis():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        response = call_periodontitis_model(temp_path)
        return jsonify(response)
    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)
