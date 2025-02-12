from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))
print("Loaded STOPWORDS:", STOPWORDS)

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Loading model and vectorizer...")  # Debugging

        predictor = pickle.load(open(r"Models/Model_XGB.pkl", "rb"))
        scaler = pickle.load(open(r"Models/Scaler.pkl", "rb"))
        cv = pickle.load(open(r"Models/ CountVectorizer.pkl", "rb"))

        print("Model and vectorizer loaded successfully.")  # Debugging

        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            print("Received a file for bulk prediction.")  # Debugging

            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif request.is_json and "text" in request.json:
            print("Received JSON data:", request.json)  # Debugging

            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            print("Predicted Sentiment:", predicted_sentiment)  # Debugging

            return jsonify({"prediction": predicted_sentiment})

        else:
            print("Invalid request format.")  # Debugging
            return jsonify({"error": "Invalid request format"}), 400

    except Exception as e:
        print("Prediction Error:", str(e))  # This will show errors in the Flask console
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)

    # Print the processed text to debug
    print("Processed Text Corpus:", corpus)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
