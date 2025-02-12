# Sentiment Analysis on Amazon Reviews

A Flask-based API that analyzes sentiment in Amazon product reviews using machine learning. It processes text data, applies TF-IDF vectorization, and uses a Logistic Regression model to classify reviews as positive, negative, or neutral.

## Features
- Preprocesses and cleans text data
- Uses TF-IDF vectorization for feature extraction
- Logistic Regression for sentiment classification
- Simple API endpoint for predictions

## How to Run

### Step 1: Clone the Repository  
```bash
git clone https://github.com/nipungoel24/Amazon-Review-Sentiment-Analysis.git
cd Amazon-Review-Sentiment-Analysis
```

### Step 2: Create and Activate a Conda Environment  
```bash
conda create -n amazonreview python=3.10
conda activate amazonreview
```

### Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask API  
```bash
flask --app api.py run
```

### Step 5: Access the API  
Once the app starts, it will run on port **5000** by default:  
```bash
http://localhost:5000
```

## API Endpoints

| Method | Endpoint  | Description |
|--------|----------|-------------|
| POST   | `/predict` | Accepts a review as input and returns its sentiment (positive/negative). |

## Future Enhancements
- Implement deep learning models for better accuracy
- Deploy the API on cloud platforms
- Improve text preprocessing techniques

## Contributions  
Feel free to fork the repo and open a pull request with your improvements!
```
