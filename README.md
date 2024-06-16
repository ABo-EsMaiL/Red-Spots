# Psoriasis Classifier

This project is a web application for classifying diseases from images using a pre-trained TensorFlow model. The application allows users to upload an image and receive a prediction with a confidence score.

## Project Structure

- `app/`: Contains the Flask application code.
- `data/`: Contains sample data and dataset.
- `models/`: Contains the saved model architecture and weights.
- `requirements.txt`: Lists the dependencies required to run the application.
- `README.md`: Project documentation.
- `.gitignore`: Specifies files to be ignored by Git.

## Setup and Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ABo-EsMaiL/Red-Spots.git
    cd Red-Spots
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the application:
    ```sh
    export FLASK_APP=app/main.py
    flask run
    ```

5. Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

1. Open the web application in your browser.
2. Upload an image.
3. View the prediction and confidence score.

## Model Information

- The model architecture is based on Xception with custom dense layers.
- The model achieves 97% accuracy on the Test set.

## License

This project is licensed under the MIT License.
