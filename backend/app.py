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
    Calls the gingivitis model on Hugging Face and returns images + analysis.
    """
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

    # Gingivitis usually returns a list of images + analysis text
    images = {}
    analysis_text = None
    labels = ["swelling", "redness", "bleeding", "diagnosis"]

    for label, item in zip(labels, result):
        if isinstance(item, str) and os.path.exists(item):
            with open(item, "rb") as f:
                images[label] = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(item, str) and (item.startswith("UklG") or item.startswith("/9j/") or len(item) > 1000):
            images[label] = item
        else:
            if "Gingivitis" in str(item) or "mean=" in str(item) or "Tooth" in str(item):
                analysis_text = item

    return {"images": images, "analysis": analysis_text}


def call_periodontitis_model(image_path, timeout_seconds=240):
    """
    Calls the periodontitis model on Hugging Face and returns flattened images + analysis.
    """
    client = Client(PERIODONTITIS_SPACE)
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
        raise TimeoutError(f"Periodontitis model timed out after {timeout_seconds}s")

    result = result_container.get("data", [])

    # Flatten lists if returned as nested lists
    flattened = []
    analysis_text = None
    if isinstance(result, tuple):
        flattened = [result[0]]
        analysis_text = result[1]
    elif isinstance(result, list):
        for r in result:
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)
    else:
        flattened = [result]

    # Encode images
    encoded_results = {}
    for i, item in enumerate(flattened):
        label = f"tooth{i+1}"
        try:
            if isinstance(item, str) and os.path.exists(item):
                with open(item, "rb") as f:
                    encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")
            else:
                encoded_results[label] = item
        except Exception as e:
            print(f"Error encoding {label}: {e}")
            encoded_results[label] = None

    return {"images": encoded_results, "analysis": analysis_text}


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
            "diagnosis": response.get("analysis") or "No diagnosis returned"
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)

