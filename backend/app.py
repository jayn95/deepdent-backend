from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GINGIVITIS_SPACE = "jayn95/deepdent_gingivitis"
PERIODONTITIS_SPACE = "jayn95/deepdent_periodontitis"


def call_huggingface(space_name, image_path, labels=None, flatten=False, timeout_seconds=120):
    """
    Calls the specified Hugging Face Space API to run a prediction with a timeout.
    Processes the result to separate analysis text from image file paths and
    encodes the images into Base64 strings.
    """
    client = Client(space_name)
    result_container = {}

    def run_predict():
        # Call the HF model with the image and confidence thresholds
        result_container["data"] = client.predict(
            handle_file(image_path),
            0.4,
            0.5,
            api_name="/predict"
        )

    # Run prediction in a separate thread with a timeout
    thread = threading.Thread(target=run_predict)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Hugging Face request to {space_name} timed out after {timeout_seconds}s")

    result = result_container.get("data", [])

    analysis_text = None
    flat_result = []

    # --- Step 1: Handle different result structures and extract analysis text ---
    if result and isinstance(result[-1], str):
        if space_name == GINGIVITIS_SPACE and "Gingivitis" in result[-1]:
            analysis_text = result[-1]
            flat_result = result[:-1]
        elif space_name == PERIODONTITIS_SPACE and ("mean=" in result[-1] or "Tooth" in result[-1]):
            analysis_text = result[-1]
            flat_result = result[:-1]
        else:
            flat_result = result
    elif isinstance(result, str):
        # If result is only a string, it's probably analysis text
        if "mean=" in result or "Tooth" in result:
            analysis_text = result
            flat_result = []
        else:
            flat_result = [result]
    else:
        flat_result = [result]

    # --- Step 2: Flatten if needed (for periodontitis, which returns lists of lists) ---
    if flatten:
        flattened = []
        for r in flat_result:
            if isinstance(r, (list, tuple)):
                flattened.extend(r)
            else:
                flattened.append(r)
        flat_result = flattened

    # --- Step 3: Generate labels for the output images ---
    if labels is None:
        labels = []
        if space_name == PERIODONTITIS_SPACE:
            # Periodontitis model returns two images per tooth (cej, abc)
            num_images = len(flat_result)
            if num_images > 0:
                num_teeth = num_images // 2
                for i in range(num_teeth):
                    for m in ["cej", "abc"]:
                        labels.append(f"tooth{i+1}_{m}")
        else:
            labels = [f"output{i+1}" for i in range(len(flat_result))]

    # --- Step 4: Encode images (Refined logic for robustness) ---
    encoded_results = {}
    for label, item in zip(labels, flat_result):
        if isinstance(item, str):
            try:
                # Case A: Gradio returned a temporary file path
                if os.path.exists(item):
                    with open(item, "rb") as f:
                        encoded_results[label] = base64.b64encode(f.read()).decode("utf-8")
                    # Note: We rely on gradio_client to handle its temp file cleanup.
                    # Explicit removal is risky as the file might already be gone.
                    
                # Case B: Gradio returned a Base64 string directly
                elif item.startswith("UklG") or item.startswith("/9j/") or len(item) > 1000:
                    encoded_results[label] = item
                else:
                    encoded_results[label] = None # String but not file or Base64
            except Exception as e:
                print(f"Error encoding image {label}: {e}")
                encoded_results[label] = None
        else:
            encoded_results[label] = None # Not a string (e.g., None, dict, number)

    # --- Step 5: Return final structure ---
    if space_name == PERIODONTITIS_SPACE:
        return {
            "images": encoded_results,
            "analysis": analysis_text
        }
    else:
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

        encoded_results = call_huggingface(
            GINGIVITIS_SPACE,
            temp_path,
            labels=["swelling", "redness", "bleeding"]
        )

        os.remove(temp_path)
        
        print(f"Gingivitis results count: {len(encoded_results)}")
        
        return jsonify({
            "images": response["images"],
            "diagnosis": encoded_results.get("analysis") or "No diagnosis returned"
        })

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

        response = call_huggingface(
            PERIODONTITIS_SPACE,
            temp_path,
            labels=None,
            flatten=True
        )

        os.remove(temp_path)

        print(f"Periodontitis images count: {len(response['images']) if response['images'] else 0}")
        print(f"Periodontitis analysis: {response.get('analysis')}")

        return jsonify({
            "images": response["images"],
            "analysis": response.get("analysis") or "No analysis text returned"
        })

    except TimeoutError as te:
        return jsonify({"error": str(te)}), 504
    except Exception as e:
        # Log the error for better debugging
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)

