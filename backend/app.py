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

    #--- Normalize result to a list
    if not isinstance(result, (list, tuple)):  #---
        result = [result]  #---

    #--- Separate text vs image paths
    analysis_texts = []  #---
    image_paths = []  #---

    for item in result:  #---
        if isinstance(item, str) and "Tooth" in item:  # likely analysis summary
            analysis_texts.append(item.strip())  #---
        elif isinstance(item, str) and os.path.exists(item):  # actual image path
            image_paths.append(item)  #---
        else:  #---
            try:  #---
                if os.path.exists(str(item)):  #---
                    image_paths.append(str(item))  #---
            except:  #---
                pass  #---

    #--- Combine text lines if multiple
    analysis_text = "\n".join(analysis_texts) if analysis_texts else ""  #---

    # Auto-generate labels
    if labels is None:
        labels = []
        if space_name == PERIODONTITIS_SPACE:
            num_teeth = len(image_paths) // 2
            for i in range(num_teeth):
                for m in ["cej", "abc"]:
                    labels.append(f"tooth{i+1}_{m}")
        else:
            labels = [f"output{i+1}" for i in range(len(image_paths))]

    # Encode available images
    encoded_results = {}
    for label, path in zip(labels, image_paths):
        if os.path.exists(path):
            with open(path, "rb") as f:
                encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")
        else:
            encoded_results[label] = None

    #--- If no images at all, return None instead of {}
    if not encoded_results:  #---
        encoded_results = None  #---

    return encoded_results, analysis_text  #---


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

        encoded_results, _ = call_huggingface(
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

        encoded_results, analysis = call_huggingface(
            PERIODONTITIS_SPACE,
            temp_path,
            labels=None,
            flatten=True
        )

        os.remove(temp_path)

        return jsonify({
            "images": encoded_results,  # may be None
            "analysis": analysis or "No analysis text returned"
        })

    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
