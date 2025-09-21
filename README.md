# Text-Sentiment-Analysis

A machine learning project that classifies tweets as positive or negative using Natural Language Processing and Logistic Regression. Built with Python and scikit-learn on the Sentiment140 dataset containing 1.6 million tweets.

🎯 Project Overview
This project demonstrates:

Text preprocessing with stemming and stopword removal
Feature extraction using TF-IDF vectorization
Machine learning classification with Logistic Regression
Model evaluation and performance metrics
Model persistence for future predictions

📊 Dataset

Source: Sentiment140 Dataset from Kaggle
Size: 1.6 million tweets
Features: 6 columns (target, id, date, flag, user, text)
Classes:

0 = Negative sentiment
1 = Positive sentiment (originally 4 in dataset)


Balance: 800,000 positive + 800,000 negative tweets

```bash
pip install -r requirements.txt
