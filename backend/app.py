from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"

# ----------------- Hugging Face Model Calls -----------------

def call_gingivitis_model(image_path, timeout_seconds=240):
    """ Calls the Hugging Face gingivitis Space and returns Base64 images and diagnosis """
    import io
    from PIL import Image
    import numpy as np

    client = Client(GINGIVITIS_SPACE)
    result_container = {}

    def run_predict():
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
    if len(result) != 4:
        raise ValueError(f"Unexpected result from gingivitis model: {result}")

    labels = ["swelling", "redness", "bleeding"]
    images = {}
    for label, item in zip(labels, result[:3]):
        try:
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "rb") as f:
                    images[label] = base64.b64encode(f.read()).decode("utf-8")
            elif isinstance(item, str):
                images[label] = item
            elif hasattr(item, "save"):
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

    diagnosis = result[3]
    return {"images": images, "diagnosis": diagnosis}


def call_periodontitis_model(image_path, timeout_seconds=240):
    """ Calls the Hugging Face periodontitis Space and returns Base64 annotated image and analysis """
    import io
    from PIL import Image
    import numpy as np
    import cv2

    client = Client(PERIODONTITIS_SPACE)
    result_container = {}

    def run_predict():
        result_container["data"] = client.predict(handle_file(image_path), api_name="/predict")

    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise TimeoutError("Periodontitis model timed out")

    data = result_container.get("data")
    if not data or len(data) != 2:
        raise ValueError(f"Unexpected result from periodontitis model: {data}")

    combined_img, summary_text = data

    if hasattr(combined_img, "save"):
        buf = io.BytesIO()
        combined_img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    elif isinstance(combined_img, np.ndarray):
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        img_b64 = base64.b64encode(buf).decode("utf-8")
    else:
        raise TypeError(f"Unexpected image type: {type(combined_img)}")

    return {"annotated_image": img_b64, "analysis": summary_text}


# ----------------- Routes -----------------

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


# Optional debug routes

@app.route("/debug/gingivitis", methods=["POST"])
def debug_gingivitis():
    image = request.files.get("image")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    client = Client(GINGIVITIS_SPACE)
    result = client.predict(handle_file(temp_path), 0.4, 0.5, api_name="/predict")
    return jsonify({"raw": str(result)})


@app.route("/debug/periodontitis", methods=["POST"])
def debug_periodontitis():
    image = request.files.get("image")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    client = Client(PERIODONTITIS_SPACE)
    result = client.predict(handle_file(temp_path), api_name="/predict")
    return jsonify({"raw": str(result)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
