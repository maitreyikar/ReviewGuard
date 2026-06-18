# Fake-Amazon-Review-Detection

This repository implements an automated approach for identifying fake reviews in Amazon product review data.

## Overview

The project evaluates whether emotional content, extracted via lexicon-based features, can improve detection of fraudulent reviews. A dense feedforward neural network (DFFNN) is used to classify reviews as legitimate or fake.

## Dataset

- Source: Amazon review dataset from Kaggle (https://www.kaggle.com/datasets/lievgarcia/amazon-reviews)
- Primary file: `datasets/amazon_reviews.txt`
- Feature-enhanced file: `datasets/amazon_reviews_features.txt`

## Feature Generation

- `Emotion_Feature_Generation.ipynb` contains the pipeline for extracting 30 emotion-based features from review text.
- These features are derived using nine emotion lexicons.
- Recomputing the feature set is not recommended unless necessary; it requires Java 1.6+ and substantial disk space and time.

## Model Training and Evaluation

- `Fake_Review_Detection.ipynb` performs preprocessing, additional feature extraction, model training, and evaluation.
- It relies on `datasets/amazon_reviews_features.txt` as input.

## Dependencies

- pandas
- nltk
- scikit-learn
- tensorflow
- gensim
- matplotlib
- re (standard library)

## Usage

1. Install the required Python packages.
2. Open `Fake_Review_Detection.ipynb` to inspect preprocessing, training, and validation.
3. Do not rerun `Emotion_Feature_Generation.ipynb` unless you intend to regenerate lexicon-based features.

## Notes

- The repository is intended for research and experimentation with fake review detection.
- The notebooks are the primary artifacts for reproducing the workflow.

