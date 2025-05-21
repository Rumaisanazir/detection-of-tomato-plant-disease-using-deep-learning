from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import joblib  # For SVM and k-NN
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import traceback

app = Flask(__name__)

# Load the ResNet50 model for feature extraction (same as in your training code)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to get a 1D feature vector
feature_extractor_model = Model(inputs=base_model.input, outputs=x)

# Ensure models are loaded
try:
    cnn_model = load_model('C:\\Users\\ANONYMOUS\\Desktop\\Project\\frontend\\cnn_model.h5')
    print("CNN Model Loaded")
except Exception as e:
    print(f"Error loading CNN model: {e}")

try:
    resnet_model = load_model('C:\\Users\\ANONYMOUS\\Desktop\\Project\\frontend\\resnet_model.h5')
    print("ResNet Model Loaded")
except Exception as e:
    print(f"Error loading ResNet model: {e}")

try:
    xception_model = load_model('C:\\Users\\ANONYMOUS\\Desktop\\Project\\frontend\\xception_model.h5')
    print("Xception Model Loaded")
except Exception as e:
    print(f"Error loading Xception model: {e}")

try:
    svm_model = joblib.load('C:\\Users\\ANONYMOUS\\Desktop\\Project\\frontend\\svm_model.pkl')
    print("SVM Model Loaded")
except Exception as e:
    print(f"Error loading SVM model: {e}")

try:
    knn_model = joblib.load('C:\\Users\\ANONYMOUS\\Desktop\\Project\\frontend\\knn_model.pkl')
    print("k-NN Model Loaded")
except Exception as e:
    print(f"Error loading k-NN model: {e}")

class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Healthy']

# Preprocessing functions
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_flattened_image(image, target_size):
    image = preprocess_image(image, target_size)
    return image.flatten().reshape(1, -1)  # Flatten the image to 1D

# Function to extract features using ResNet50
def extract_features(image):
    try:
        image_resized = image.resize((224, 224))  # Resize image to 224x224
        image_array = np.array(image_resized) / 255.0  # Normalize image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        features = feature_extractor_model.predict(image_array)
        print(f"Extracted features shape: {features.shape}")  # Debugging feature shape
        return features.flatten()  # Flatten the output to a 1D array
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image
        image = Image.open(file)

        # Preprocess image for CNN, ResNet, and Xception (all expect 224x224x3)
        cnn_input = preprocess_image(image, (224, 224))
        resnet_input = preprocess_image(image, (224, 224))
        xception_input = preprocess_image(image, (224, 224))

        # Preprocess image for SVM and k-NN (use feature extraction for SVM and k-NN)
        svm_input = extract_features(image)
        knn_input = extract_features(image)

        # Log the shape of the inputs for debugging
        if svm_input is not None:
            print(f"SVM input shape: {svm_input.shape}, Sample: {svm_input[:10]}")
        if knn_input is not None:
            print(f"k-NN input shape: {knn_input.shape}, Sample: {knn_input[:10]}")

        # Predict using each model
        results = []

        # CNN prediction
        try:
            cnn_pred = cnn_model.predict(cnn_input)
            results.append({
                'model': 'CNN',
                'predicted_class': class_names[np.argmax(cnn_pred)],
                'confidence': float(np.max(cnn_pred)),
                'accuracy': 0.87
            })
        except Exception as e:
            print(f"Error in CNN prediction: {e}")
        
        # ResNet prediction
        try:
            resnet_pred = resnet_model.predict(resnet_input)
            results.append({
                'model': 'ResNet',
                'predicted_class': class_names[np.argmax(resnet_pred)],
                'confidence': float(np.max(resnet_pred)),
                'accuracy': 0.91
            })
        except Exception as e:
            print(f"Error in ResNet prediction: {e}")

        # Xception prediction
        try:
            xception_pred = xception_model.predict(xception_input)
            results.append({
                'model': 'Xception',
                'predicted_class': class_names[np.argmax(xception_pred)],
                'confidence': float(np.max(xception_pred)),
                'accuracy': 0.93
            })
        except Exception as e:
            print(f"Error in Xception prediction: {e}")

        # SVM prediction
        try:
            if svm_input is not None:
                svm_pred = svm_model.predict([svm_input])  # Ensure input is 2D
                svm_pred_class = int(svm_pred[0])  # Convert to integer
                results.append({
                    'model': 'SVM',
                    'predicted_class': class_names[svm_pred_class],
                    'confidence': 1.0,
                    'accuracy': 0.51
                })
        except Exception as e:
            print(f"Error in SVM prediction: {e}")
            traceback.print_exc()

        # k-NN prediction
        try:
            if knn_input is not None:
                knn_pred = knn_model.predict([knn_input])  # Ensure input is 2D
                knn_pred_class = int(knn_pred[0])   # Convert to integer
                results.append({
                    'model': 'k-NN',
                    'predicted_class': class_names[knn_pred_class],
                    'confidence': 1.0,
                    'accuracy': 0.43
                })
        except Exception as e:
            print(f"Error in k-NN prediction: {e}")
            traceback.print_exc()

        # Ensemble Voting: Combine predictions from all models
        all_classes = [result['predicted_class'] for result in results]
        final_prediction = max(set(all_classes), key=all_classes.count)
        final_confidence = max([result['confidence'] for result in results if result['predicted_class'] == final_prediction])

        # Add Ensemble Voting result to the results
        ensemble_result = {
            'model': 'Ensemble Voting',
            'predicted_class': final_prediction,
            'confidence': final_confidence,
            'accuracy': 0.85
        }

        results.append(ensemble_result)

        # Return results in JSON format
        return jsonify({'results': results})

    except Exception as e:
        print("Error during prediction:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  