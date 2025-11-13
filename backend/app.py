from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"


def call_huggingface(space_name, image_path, labels=None, flatten=False, timeout_seconds=120):
    client = Client(space_name)
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
        raise TimeoutError(f"Hugging Face request to {space_name} timed out after {timeout_seconds}s")

    result = result_container.get("data", [])

    analysis_text = None
    flat_result = []

    if isinstance(result, (list, tuple)):
        if isinstance(result[-1], str) and len(result) > 1:
            analysis_text = result[-1]
            flat_result = result[:-1]
        else:
            flat_result = result
    elif isinstance(result, str):
        analysis_text = result
    else:
        flat_result = [result]

    if flatten:
        flattened = []
        for r in flat_result:
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)
        flat_result = flattened

    if labels is None:
        labels = []
        if space_name == PERIODONTITIS_SPACE:
            num_teeth = len(flat_result) // 2
            for i in range(num_teeth):
                for m in ["cej", "abc"]:
                    labels.append(f"tooth{i+1}_{m}")
        else:
            labels = [f"output{i+1}" for i in range(len(flat_result))]

    encoded_results = {}
    for label, item in zip(labels, flat_result):
        if isinstance(item, str):
            #-- Case 1: it's a path
            if os.path.exists(item):
                with open(item, "rb") as f:
                    encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")
            #-- Case 2: already Base64 or data URL
            elif item.strip().startswith("UklG") or item.strip().startswith("/9j/"):
                encoded_results[label] = item
        else:
            encoded_results[label] = None

    #-- Return structure depends on the space
    if space_name == PERIODONTITIS_SPACE:
        return {
            "images": encoded_results,
            "analysis": analysis_text
        }
    else:
        #-- Gingivitis returns only images
        return encoded_results


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

        #-- Gingivitis: returns only images (original behavior)
        encoded_results = call_huggingface(
            GINGIVITIS_SPACE,
            temp_path,
            labels=["swelling", "redness", "bleeding"]
        )

        os.remove(temp_path)
        return jsonify({"images": encoded_results})

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

        #-- Periodontitis: returns both images and analysis
        response = call_huggingface(
            PERIODONTITIS_SPACE,
            temp_path,
            labels=None,
            flatten=True
        )

        os.remove(temp_path)

        return jsonify({
            "images": response["images"],
            "analysis": response.get("analysis") or "No analysis text returned"
        })

    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)