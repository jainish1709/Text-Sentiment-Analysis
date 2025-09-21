# Twitter Sentiment Analysis

A machine learning project that classifies tweets as **positive** or **negative** using natural language processing and logistic regression.

## 🎯 Project Overview

This project analyzes the sentiment of Twitter data using the famous **Sentiment140 dataset** containing 1.6 million tweets. The model achieves **77.8% accuracy** in predicting whether a tweet expresses positive or negative sentiment.

## ✨ Features

- **Large Dataset**: Trained on 1.6 million tweets from Sentiment140
- **Text Preprocessing**: Advanced cleaning with stemming and stopword removal  
- **Machine Learning**: Logistic Regression model with TF-IDF vectorization
- **High Accuracy**: 77.8% accuracy on test data
- **Saved Model**: Pre-trained model ready for predictions

## 📊 Dataset

- **Source**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle
- **Size**: 1,600,000 tweets
- **Classes**: 
  - `0` = Negative sentiment
  - `1` = Positive sentiment
- **Features**: Tweet text, user, timestamp, and sentiment labels

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jainish1709/Text-Sentiment-Analysis.git
   cd Text-Sentiment-Analysis
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (first time only)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Dataset Setup

1. **Get Kaggle API credentials**
   - Go to Kaggle → Account → API → Create New API Token
   - Download `kaggle.json` file

2. **Download the dataset**
   ```bash
   kaggle datasets download -d kazanova/sentiment140
   ```

3. **Extract the dataset**
   - The notebook will automatically extract the CSV file

## 📖 Usage

### Training the Model

1. Open `Text Sentiment Analysis.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The model will be automatically saved as `trained_model.sav`

### Making Predictions

```python
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('trained_model.sav', 'rb'))

# Make prediction on new text
# (Note: text needs to be preprocessed and vectorized same way as training data)
```

## 📈 Model Performance

| Metric | Training Data | Test Data |
|--------|---------------|-----------|
| **Accuracy** | 79.87% | 77.67% |
| **Dataset Size** | 1,280,000 tweets | 320,000 tweets |

## 🛠️ Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Development environment

## 📁 Project Structure

```
Text-Sentiment-Analysis/
├── Text Sentiment Analysis.ipynb    # Main notebook with complete analysis
├── trained_model.sav                # Saved logistic regression model
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file
```

## 🔍 Methodology

1. **Data Loading**: Import 1.6M tweets from Sentiment140 dataset
2. **Preprocessing**: 
   - Remove URLs, mentions, special characters
   - Convert to lowercase
   - Remove stopwords
   - Apply Porter Stemming
3. **Feature Extraction**: TF-IDF Vectorization
4. **Model Training**: Logistic Regression with 80-20 train-test split
5. **Evaluation**: Accuracy metrics on test data

## 🚧 Future Improvements

- [ ] Try advanced models (Random Forest, SVM, Neural Networks)
- [ ] Implement cross-validation for better model evaluation
- [ ] Add confusion matrix and detailed classification metrics
- [ ] Create a web interface for real-time predictions
- [ ] Expand to multi-class sentiment (positive, negative, neutral)
- [ ] Add support for other social media platforms

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.6
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
kaggle>=1.5.0
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Jainish** 
- GitHub: [@jainish1709](https://github.com/jainish1709)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/jainish1709/Text-Sentiment-Analysis/issues).

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

*Built with ❤️ using Python and Machine Learning*
