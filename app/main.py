from flask import Flask, request, render_template
from app.model import load_model, predict_image
from app.utils import preprocess_image

app = Flask(__name__)
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img = preprocess_image(file)
            class_name, confidence = predict_image(model, img)
            return render_template('index.html', class_name=class_name, confidence=confidence)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
