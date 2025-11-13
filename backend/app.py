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

    #-- Handle different result structures
    if isinstance(result, (list, tuple)):
        #-- Check if last item is analysis text
        if result and isinstance(result[-1], str) and ("mean=" in result[-1] or "Tooth" in result[-1]):
            analysis_text = result[-1]
            flat_result = result[:-1]
        else:
            flat_result = result
    elif isinstance(result, str):
        #-- If result is only a string, it's probably analysis text
        if "mean=" in result or "Tooth" in result:
            analysis_text = result
            flat_result = []
        else:
            flat_result = [result]
    else:
        flat_result = [result]

    #-- Flatten if needed
    if flatten:
        flattened = []
        for r in flat_result:
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)
        flat_result = flattened

    #-- Generate labels
    if labels is None:
        labels = []
        if space_name == PERIODONTITIS_SPACE:
            num_teeth = len(flat_result) // 2
            for i in range(num_teeth):
                for m in ["cej", "abc"]:
                    labels.append(f"tooth{i+1}_{m}")
        else:
            labels = [f"output{i+1}" for i in range(len(flat_result))]

    #-- Encode images
    encoded_results = {}
    for label, item in zip(labels, flat_result):
        if isinstance(item, str):
            #-- Case 1: it's a file path
            if os.path.exists(item):
                with open(item, "rb") as f:
                    encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")
            #-- Case 2: already Base64 string
            elif item.startswith("UklG") or item.startswith("/9j/") or len(item) > 1000:
                encoded_results[label] = item
        elif item is not None:
            encoded_results[label] = None

    #-- For periodontitis: return both images and analysis
    if space_name == PERIODONTITIS_SPACE:
        return {
            "images": encoded_results,
            "analysis": analysis_text
        }
    else:
        #-- For gingivitis: return only images
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

        #-- Gingivitis returns only images
        encoded_results = call_huggingface(
            GINGIVITIS_SPACE,
            temp_path,
            labels=["swelling", "redness", "bleeding"]  #-- Should return 3 images
        )

        os.remove(temp_path)
        
        #-- Debug: Check how many images we got
        print(f"Gingivitis results count: {len(encoded_results)}")
        for key in encoded_results:
            print(f" - {key}: {'exists' if encoded_results[key] else 'None'}")
        
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

        #-- Periodontitis returns both images and analysis
        response = call_huggingface(
            PERIODONTITIS_SPACE,
            temp_path,
            labels=None,
            flatten=True
        )

        os.remove(temp_path)

        #-- Debug: Check what we received
        print(f"Periodontitis images count: {len(response['images']) if response['images'] else 0}")
        print(f"Periodontitis analysis: {response.get('analysis')}")

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