from flask import Flask, request, jsonify, render_template
from features_management import extract_features
from compare_features import find_similar_images

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image_path = f'uploads/{file.filename}'
    file.save(image_path)

    # Extract features from the uploaded image
    uploaded_image_features = extract_features(image_path)

    # Find similar images in the dataset
    similar_images = find_similar_images(uploaded_image_features, 'dataset/')

    return jsonify(similar_images)

if __name__ == '__main__':
    app.run(debug=True)
