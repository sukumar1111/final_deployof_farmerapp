from flask import Flask, request, render_template, send_from_directory
import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, send_from_directory, url_for

app = Flask(__name__)

# Disable GPU usage for TensorFlow Lite
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the TFLite model using TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path="lastestminimodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image
def preprocess_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        return None  # Handle cases where OpenCV fails to read the image

    image = cv.resize(image, (128, 128))  # Resize image to match model input size
    # image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.reshape(image, (1, 128, 128, 3))  # Reshape to match input tensor shape
    return image

# Function to make predictions using TFLite model
def predict_tflite(image):
    # Ensure input shape matches model's expected shape
    input_data = np.array(image, dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
CLASS_LABELS = {
    0: {
        "disease": "Apple___Apple_scab",
        "cause": "Caused by the fungus Venturia inaequalis, results in dark lesions on leaves and fruit.",
        "pesticide": "Captan, Mancozeb",
        "usage": "Apply fungicides during early leaf development. Practice crop rotation and remove infected debris."
    },
    1: {
        "disease": "Apple___Black_rot",
        "cause": "Caused by the fungus Botryosphaeria obtusa, leads to fruit rot and leaf spots.",
        "pesticide": "Thiophanate-methyl, Captan",
        "usage": "Prune and remove cankers. Spray fungicide during bloom and petal fall."
    },
    2: {
        "disease": "Apple___Cedar_apple_rust",
        "cause": "Caused by the fungus Gymnosporangium juniperi-virginianae, creates bright orange spots on leaves.",
        "pesticide": "Myclobutanil",
        "usage": "Apply fungicide early in the season. Remove nearby juniper hosts."
    },
    3: {
        "disease": "Apple___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Maintain regular watering and pruning practices."
    },
    4: {
        "disease": "Blueberry___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Keep soil acidic and ensure proper drainage."
    },
    5: {
        "disease": "Cherry_(including_sour)___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Ensure adequate sunlight and monitor for pests."
    },
    6: {
        "disease": "Cherry_(including_sour)___Powdery_mildew",
        "cause": "Caused by Podosphaera clandestina, results in white powdery growth on leaves.",
        "pesticide": "Sulfur-based fungicides",
        "usage": "Apply at early bloom and repeat every 10–14 days as needed."
    },
    7: {
        "disease": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "cause": "Caused by Cercospora zeae-maydis, appears as gray to tan streaks on leaves.",
        "pesticide": "Strobilurin, Triazole",
        "usage": "Use resistant hybrids and apply fungicides at tasseling if disease pressure is high."
    },
    8: {
        "disease": "Corn_(maize)___Common_rust_",
        "cause": "Caused by Puccinia sorghi, forms reddish-brown pustules on leaves.",
        "pesticide": "Fungicides like Azoxystrobin",
        "usage": "Apply fungicide when rust is first observed in the field."
    },
    9: {
        "disease": "Corn_(maize)___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Use disease-resistant hybrids and ensure proper fertilization."
    },
    10: {
        "disease": "Corn_(maize)___Northern_Leaf_Blight",
        "cause": "Caused by Exserohilum turcicum, creates long grayish lesions on leaves.",
        "pesticide": "Triazoles, Strobilurins",
        "usage": "Apply fungicides at early tassel stage if disease incidence is high."
    },
    11: {
        "disease": "Grape___Black_rot",
        "cause": "Caused by Guignardia bidwellii, forms black spots on leaves and shriveled fruits.",
        "pesticide": "Mancozeb, Captan",
        "usage": "Spray fungicides from early shoot growth and maintain regular schedule."
    },
    12: {
        "disease": "Grape___Esca_(Black_Measles)",
        "cause": "Caused by a complex of fungi including Phaeomoniella, leads to leaf spotting and fruit rotting.",
        "pesticide": "No effective chemical control",
        "usage": "Remove infected vines and maintain good vineyard sanitation."
    },
    13: {
        "disease": "Grape___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Ensure good airflow and avoid excessive irrigation."
    },
    14: {
        "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "cause": "Caused by Pseudocercospora vitis, causes angular brown lesions on leaves.",
        "pesticide": "Chlorothalonil, Copper-based fungicides",
        "usage": "Start spraying at early growth stages and repeat as needed."
    },
    15: {
        "disease": "Orange___Haunglongbing_(Citrus_greening)",
        "cause": "Caused by Candidatus Liberibacter asiaticus, spread by psyllids.",
        "pesticide": "No cure, manage psyllid vector",
        "usage": "Control psyllids and remove infected trees. Monitor regularly."
    },
    16: {
        "disease": "Peach___Bacterial_spot",
        "cause": "Caused by Xanthomonas arboricola, results in lesions on leaves and fruit.",
        "pesticide": "Copper-based sprays",
        "usage": "Apply during early leaf stage and avoid overhead watering."
    },
    17: {
        "disease": "Peach___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Ensure balanced nutrition and avoid leaf wetness."
    },
    18: {
        "disease": "Pepper__bell___Bacterial_spot",
        "cause": "Caused by Xanthomonas campestris, shows water-soaked spots on leaves.",
        "pesticide": "Copper-based fungicides",
        "usage": "Spray at first sign and practice crop rotation."
    },
    19: {
        "disease": "Pepper__bell___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Water at base and keep foliage dry to prevent diseases."
    },
    20: {
        "disease": "Potato___Early_blight",
        "cause": "Caused by Alternaria solani, produces concentric rings on older leaves.",
        "pesticide": "Chlorothalonil, Mancozeb",
        "usage": "Spray fungicide during humid conditions and remove infected foliage."
    },
    21: {
        "disease": "Potato___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Ensure proper spacing and drainage to avoid disease buildup."
    },
    22: {
        "disease": "Potato___Late_blight",
        "cause": "Caused by Phytophthora infestans, leads to large, dark, rapidly spreading lesions.",
        "pesticide": "Metalaxyl, Mancozeb",
        "usage": "Apply fungicides at first sign. Remove infected plants promptly."
    },
    23: {
        "disease": "Raspberry___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Maintain airflow and remove dead canes to prevent disease."
    },
    24: {
        "disease": "Soybean___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Use certified seeds and practice crop rotation."
    },
    25: {
        "disease": "Squash___Powdery_mildew",
        "cause": "Caused by Podosphaera xanthii, leaves show white powdery patches.",
        "pesticide": "Sulfur, Potassium bicarbonate",
        "usage": "Apply early and reapply every 7–10 days if needed."
    },
    26: {
        "disease": "Strawberry___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Keep beds weed-free and water early in the day."
    },
    27: {
        "disease": "Strawberry___Leaf_scorch",
        "cause": "Caused by Diplocarpon earliana, produces purplish spots on leaves.",
        "pesticide": "Myclobutanil, Captan",
        "usage": "Apply fungicide in spring and maintain dry leaf surfaces."
    },
    28: {
        "disease": "Tomato___Bacterial_spot",
        "cause": "Caused by Xanthomonas bacteria, leading to dark, water-soaked spots on leaves and fruits.",
        "pesticide": "Copper-based fungicides",
        "usage": "Apply preventively or at the first sign of infection. Ensure good air circulation and avoid overhead watering."
    },
    29: {
        "disease": "Tomato___Early_blight",
        "cause": "Caused by Alternaria solani, leading to brown concentric rings on lower leaves.",
        "pesticide": "Chlorothalonil, Mancozeb",
        "usage": "Apply fungicides at regular intervals during humid conditions. Remove infected leaves to reduce spread."
    },
    30: {
        "disease": "Tomato___healthy",
        "cause": "No disease present.",
        "pesticide": "No pesticide needed",
        "usage": "Maintain proper plant care, including watering and fertilization, to prevent diseases."
    },
    31: {
        "disease": "Tomato___Late_blight",
        "cause": "Caused by Phytophthora infestans, causing large, irregular brown patches.",
        "pesticide": "Metalaxyl, Copper fungicides",
        "usage": "Spray preventively, especially in wet conditions. Remove infected plants."
    },
    32: {
        "disease": "Tomato___Leaf_Mold",
        "cause": "Caused by Passalora fulva, appears as yellow spots and fuzzy mold underneath leaves.",
        "pesticide": "Chlorothalonil, Copper-based",
        "usage": "Ensure airflow in greenhouse. Apply fungicides early in disease development."
    },
    33: {
        "disease": "Tomato___Septoria_leaf_spot",
        "cause": "Caused by Septoria lycopersici, shows small round spots with dark borders.",
        "pesticide": "Mancozeb, Chlorothalonil",
        "usage": "Remove infected foliage and apply fungicide every 7–10 days."
    },
    34: {
        "disease": "Tomato___Spider_mites Two-spotted_spider_mite",
        "cause": "Infestation by Tetranychus urticae, causes stippling and webbing on leaves.",
        "pesticide": "Miticides like Abamectin",
        "usage": "Spray thoroughly on undersides of leaves. Isolate infested plants."
    },
    35: {
        "disease": "Tomato___Target_Spot",
        "cause": "Caused by Corynespora cassiicola, leads to concentric lesions on leaves.",
        "pesticide": "Chlorothalonil, Mancozeb",
        "usage": "Apply fungicide at first sign. Ensure good spacing and airflow."
    },
    36: {
        "disease": "Tomato___Tomato_mosaic_virus",
        "cause": "Spread mechanically or by infected tools, causes mottling and leaf curling.",
        "pesticide": "No cure",
        "usage": "Remove infected plants and disinfect tools regularly."
    },
    37: {
        "disease": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "cause": "Spread by whiteflies, causes yellowing and curling of leaves.",
        "pesticide": "Insecticidal soaps, Imidacloprid",
        "usage": "Control whiteflies and remove infected plants."
    }
}



# Flask app routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            image = preprocess_image(file_path)
            if image is None:
                return render_template("index.html", error="Error: Invalid image file")

            prediction = predict_tflite(image)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Ensure CLASS_LABELS includes pesticide details
            result = CLASS_LABELS.get(predicted_class, {"disease": "Unknown","cause":"unknown", "usage":"Unkown","pesticide": "Unknown"})
            
            pesticide_name = result.get("pesticide", "Unknown")
            pesticide_image = url_for('static', filename=f'pesticides/{predicted_class}.jpg') if predicted_class <=38 else None

            return render_template("index.html", filename=filename, 
                                   disease=result["disease"], cause=result["cause"],pesticide=pesticide_name, 
                                    
                                   pesticide_image=pesticide_image,usage=result["usage"])

    return render_template("index.html")




@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
