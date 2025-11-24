from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"


# --- Hugging Face Model Calls ---

def call_gingivitis_model(image_path, timeout_seconds=240):
    client = Client(GINGIVITIS_SPACE)
    result_container = {}

    def run_predict():
        try:
            result_container["data"] = client.predict(
                handle_file(image_path),
                0.4,
                0.5,
                api_name="/predict"
            )
        except Exception as e:
            result_container["error"] = str(e)

    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Gingivitis model timed out after {timeout_seconds}s")

    if "error" in result_container:
        raise RuntimeError(result_container["error"])

    result = result_container.get("data")
    if not result or len(result) != 4:
        raise ValueError(f"Unexpected result from gingivitis model: {result}")

    labels = ["swelling", "redness", "bleeding"]
    images = {}

    for label, item in zip(labels, result[:3]):
        try:
            if isinstance(item, str):
                if os.path.exists(item):
                    with open(item, "rb") as f:
                        images[label] = base64.b64encode(f.read()).decode("utf-8")
                else:
                    images[label] = item
            else:
                import io
                from PIL import Image
                import numpy as np
                if hasattr(item, "save"):
                    buf = io.BytesIO()
                    item.save(buf, format="JPEG")
                    images[label] = base64.b64encode(buf.getvalue()).decode("utf-8")
                elif isinstance(item, np.ndarray):
                    pil_img = Image.fromarray(item)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    images[label] = base64.b64encode(buf.getvalue()).decode("utf-8")
                else:
                    images[label] = None
        except Exception as e:
            print(f"Error converting {label} to Base64: {e}")
            images[label] = None

    diagnosis = result[3] or "No diagnosis returned"
    return {"images": images, "diagnosis": diagnosis}


def call_periodontitis_model(image_path, timeout_seconds=240):
    client = Client(PERIODONTITIS_SPACE)
    result_container = {}

    def run_predict():
        try:
            result_container["data"] = client.predict(
                handle_file(image_path),
                api_name="/predict"
            )
        except Exception as e:
            result_container["error"] = str(e)

    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError("Periodontitis model timed out")

    if "error" in result_container:
        raise RuntimeError(result_container["error"])

    data = result_container.get("data")
    print("HF Periodontitis result:", data)

    if not data or len(data) < 2:
        raise ValueError(f"Unexpected HF Periodontitis output: {data}")

    combined_img = data[0]
    summary_text = data[1]

    import numpy as np
    import cv2

    if isinstance(combined_img, np.ndarray):
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        img_b64 = base64.b64encode(buf).decode("utf-8")
    else:
        # Assume HF already returned Base64 string
        img_b64 = combined_img

    return {"annotated_image": img_b64, "analysis": summary_text}


# --- Routes ---

@app.route("/")
def home():
    return jsonify({"status": "DeepDent backend running successfully!"})


@app.route("/predict/gingivitis", methods=["POST"])
def predict_gingivitis():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        response = call_gingivitis_model(temp_path)
        return jsonify({
            "images": response["images"],
            "diagnosis": response.get("diagnosis")
        })
    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        print("Gingivitis error:", e)
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
        print("Periodontitis error:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
