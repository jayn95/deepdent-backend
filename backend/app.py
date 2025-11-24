from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile, base64, os, threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# NEW — Separate Spaces
SWELLING_SPACE = "DeepdentTeam/Gingiviitis-Swelling"
REDNESS_SPACE = "DeepdentTeam/Gingiviitis-Redness"
BLEEDING_SPACE = "DeepdentTeam/Gingiviitis-Bleeding"

PERIODONTITIS_SPACE = "DeepdentTeam/deepdent_periodontitis"

def call_huggingface(space_name, image_path, labels=None, flatten=False, timeout_seconds=300):
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
    
    # Handle tuple case (periodontitis returns single image + analysis)
    if isinstance(result, tuple):
        flat_result = [result[0]]
        analysis_text = result[1]
    else:
        # Fallback for list or single string (gingivitis)
        if isinstance(result, list):
            flat_result = []
            analysis_text = None
            for item in result:
                if isinstance(item, str) and ("Gingivitis" in item or "mean=" in item or "Tooth" in item):
                    analysis_text = item
                else:
                    flat_result.append(item)
        elif isinstance(result, str):
            flat_result = []
            if "mean=" in result or "Tooth" in result or "Gingivitis" in result:
                analysis_text = result
            else:
                flat_result = [result]
        else:
            flat_result = [result]
            analysis_text = None

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
        if space_name == PERIODONTITIS_SPACE:
            labels = [f"tooth1"]
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
            # NEW: HF may return numpy arrays (images)
            try:
                import numpy as np
                from PIL import Image
                import io
        
                if isinstance(item, np.ndarray):
                    # convert numpy → PIL
                    pil_img = Image.fromarray(item)
        
                    # save to bytes
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG")
                    encoded_results[label] = base64.b64encode(buf.getvalue()).decode("utf-8")
                else:
                    encoded_results[label] = None
        
            except Exception as e:
                print(f"Error encoding numpy image {label}: {e}")
                encoded_results[label] = None

    # --- Step 5: Return final structure ---
    # --- Step 5: Return final structure ---
    if space_name == PERIODONTITIS_SPACE:
        return {
            "images": encoded_results,
            "analysis": analysis_text
        }
    else:
        return encoded_results

def call_single_model(space_name, image_path):
    """
    Calls a Hugging Face Space that returns ONLY:
    [image, diagnosis_text]
    """
    client = Client(space_name)

    result = client.predict(
        handle_file(image_path),
        0.4,
        0.5,
        api_name="/predict"
    )

    # Unpack results
    image_item = result[0]
    diagnosis_text = result[1]

    # Convert to base64
    encoded = None

    try:
        # Case A: path to temp file
        if isinstance(image_item, str) and os.path.exists(image_item):
            with open(image_item, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

        # Case B: numpy array
        else:
            import numpy as np
            if isinstance(image_item, np.ndarray):
                from PIL import Image
                import io
                pil_img = Image.fromarray(image_item)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print("Error encoding image:", e)

    return encoded, diagnosis_text


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

        # Call the 3 individual Spaces
        swell_img, swell_diag = call_single_model(SWELLING_SPACE, temp_path)
        red_img, red_diag = call_single_model(REDNESS_SPACE, temp_path)
        bleed_img, bleed_diag = call_single_model(BLEEDING_SPACE, temp_path)

        os.remove(temp_path)

        return jsonify({
            "images": {
                "swelling": swell_img,
                "redness": red_img,
                "bleeding": bleed_img
            },
            "diagnosis": {
                "swelling": swell_diag,
                "redness": red_diag,
                "bleeding": bleed_diag
            }
        })

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

@app.route("/debug/periodontitis", methods=["POST"])
def debug_periodontitis():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        client = Client(PERIODONTITIS_SPACE)
        result = client.predict(
            handle_file(temp_path),
            0.4,
            0.5,
            api_name="/predict"
        )

        print("RAW PERIODONTITIS RESULT:", result)
        return jsonify({"raw": str(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)












