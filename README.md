# Tweet Sentiment Analyzer

A small machine learning app that reads a tweet and tells you whether the sentiment is positive, negative or neutral, along with how confident the model is. It uses TF-IDF features and logistic regression, and the whole thing runs behind a simple Gradio page, so you can paste a tweet and see the result in a second.

I built this to go through the full cycle once: clean raw text, train a classifier, save it, and put it in front of a real interface instead of leaving it in a notebook.

## How it works

There are two separate parts, and they only meet through two saved files.

Training happens in `trainmodel.py`. It reads the training CSV, cleans the tweet text, fits a TF-IDF vectorizer, splits the data into train and test sets, trains a logistic regression model, and prints an evaluation on the held-out test set. At the end it saves the model to `model.pkl` and the vectorizer to `vectorizer.pkl`.

Prediction happens in `app.py`. It loads those two pickle files, applies the same text cleaning used in training, transforms the tweet with the saved vectorizer, and asks the model for a class and a probability. The class is mapped to a readable label and shown in the Gradio interface together with the confidence score.

Keeping the cleaning step identical on both sides matters. If the app cleans text differently from training, the vectorizer sees words it never learned and predictions get worse without any error telling you why.

A full diagram of the flow is in `diagram.png`.

## Project structure

```
.
├── app.py             # Gradio app, prediction logic, label mapping
├── trainmodel.py      # cleaning, TF-IDF, split, training, evaluation
├── model.pkl          # trained logistic regression model
├── vectorizer.pkl     # fitted TF-IDF vectorizer
├── requirements.txt
└── README.md
```

The training CSV is not included in the repo. Put it in the project folder and point `trainmodel.py` at it (see below).

## Getting started

Clone the repo and install the dependencies:

```bash
git clone https://github.com/minahilirfan98/datascience-ai.git
cd datascience-ai
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

Make sure the path to your CSV at the top of `trainmodel.py` is correct, then run:

```bash
python trainmodel.py
```

When it finishes you will see the evaluation results in the terminal and two new files, `model.pkl` and `vectorizer.pkl`, in the project folder. If you only want to try the app, you can skip this step when those files are already in the repo.

### Run the app

```bash
python app.py
```

Gradio prints a local address, usually `http://127.0.0.1:7860`. Open it in your browser, type or paste a tweet, and submit.

## Example

Input:

```
Just got my results back and I passed everything, couldn't be happier
```

Output:

```
Sentiment: Positive
Confidence: 0.94
```

Your exact confidence numbers will depend on the data you trained on.

## Tech stack

Python, scikit-learn (TF-IDF and logistic regression), pandas for loading the data, and Gradio for the interface.

## Limitations

The model is a bag-of-words classifier, so it looks at which words appear and not at how they are used. Sarcasm, negation ("not bad at all") and slang it has never seen will trip it up. It is also only as good as the labels in the training CSV, and tweets in languages other than the one it was trained on will not give meaningful results.

## Ideas for next steps

Compare logistic regression against a linear SVM or Naive Bayes on the same features, add n-grams to the vectorizer to catch short phrases, and try a small transformer model to see how much the extra context helps. Deploying the app on Hugging Face Spaces would also make it easy to share a live demo.

## Author

Minahil Irfan
Portfolio: [behance.net/minahilirfan2](https://www.behance.net/minahilirfan2)
