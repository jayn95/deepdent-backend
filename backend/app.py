from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ Allow mobile apps to call this API

# Hugging Face Spaces
GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"


def call_huggingface(space_name, image_path, labels=None, flatten=False, timeout_seconds=120):
    """Calls Hugging Face Space in a background thread with timeout."""
    client = Client(space_name)
    result_container = {}

    def run_predict():
        # some models expect confidence params (e.g., 0.4, 0.5)
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
        raise TimeoutError(f"Hugging Face request to {space_name} timed out after {timeout_seconds}s")

    result = result_container.get("data", [])

    # --- NEW HYBRID HANDLER ---
    analysis_text = None
    flat_result = []

    # If it's a list or tuple, inspect contents
    if isinstance(result, (list, tuple)):
        # Detect if last item is text
        if isinstance(result[-1], str) and len(result) > 1:
            analysis_text = result[-1]
            flat_result = result[:-1]
        else:
            flat_result = result
    elif isinstance(result, str):
        # Only text result
        analysis_text = result
    else:
        flat_result = [result]

    # Optionally flatten nested lists
    if flatten:
        flattened = []
        for r in flat_result:
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)
        flat_result = flattened

    # Generate default labels if not provided
    if labels is None:
        labels = []
        if space_name == PERIODONTITIS_SPACE:
            num_teeth = len(flat_result) // 2
            for i in range(num_teeth):
                for m in ["cej", "abc"]:
                    labels.append(f"tooth{i+1}_{m}")
        else:
            labels = [f"output{i+1}" for i in range(len(flat_result))]

    # Encode valid images
    encoded_results = {}
    for label, path in zip(labels, flat_result):
        if isinstance(path, str) and os.path.exists(path):
            with open(path, "rb") as f:
                encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")

    return {
        "images": encoded_results,
        "analysis": analysis_text
    }


@app.route("/")
def home():
    return jsonify({"status": "DeepDent backend running successfully!"})


@app.route("/predict/gingivitis", methods=["POST"])
def predict_gingivitis():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        result = call_huggingface(
            GINGIVITIS_SPACE,
            temp_path,
            labels=["swelling", "redness", "bleeding"]
        )

        os.remove(temp_path)
        return jsonify(result)

    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/periodontitis", methods=["POST"])
def predict_periodontitis():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        result = call_huggingface(
            PERIODONTITIS_SPACE,
            temp_path,
            labels=None,
            flatten=True
        )

        os.remove(temp_path)
        return jsonify(result)

    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
