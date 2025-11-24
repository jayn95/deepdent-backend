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
    Calls the periodontitis HF Space and returns:
    - Base64 encoded images
    - Structured measurements
    - Summary text
    """
    client = Client(PERIODONTITIS_SPACE)
    result_container = {}

    def run_predict():
        result_container["data"] = client.predict(
            handle_file(image_path),
            0.4,    # threshold 1
            0.5,    # threshold 2
            api_name="/predict"
        )

    # Run with timeout
    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Periodontitis model timed out after {timeout_seconds}s")

    # HF Results
    result = result_container.get("data", None)

    # =====================================================
    # Expected HF return format:
    # (combined_image_path, measurements_list, summary_text)
    # =====================================================
    print("🔍 HF RAW RESULT:", result, type(result))

    if not isinstance(result, list) or len(result) != 3:
        raise ValueError("Unexpected model output format from HF space")

    combined_image_path = result[0]
    measurements = result[1]
    summary_text = result[2]

    # =====================================================
    # Encode the combined output image as base64
    # =====================================================
    encoded_images = {}

    try:
        if isinstance(combined_image_path, str) and os.path.exists(combined_image_path):
            with open(combined_image_path, "rb") as f:
                encoded_images["combined"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            encoded_images["combined"] = combined_image_path
    except Exception as e:
        print(f"Error encoding combined image: {e}")
        encoded_images["combined"] = None

    # =====================================================
    # Final response to Flutter
    # =====================================================
    return {
        "images": encoded_images,
        "measurements": measurements,   # <---- structured list
        "summary_text": summary_text    # <---- readable text
    }


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
        return jsonify({
            "images": response["images"],
            "analysis": response.get("analysis") or "No analysis text returned"
        })
    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)



