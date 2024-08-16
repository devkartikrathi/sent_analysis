from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    text_input = request.form["text"] 
    # text_input = "I love this product! It's amazing!"
    # text_input = "Does not work all the time"
    print(text_input)
    predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
    return jsonify({"prediction": predicted_sentiment})

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)