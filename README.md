# Spam Mail Prediction System

This repository contains a **Spam Mail Prediction System** that classifies emails as **Spam** or **Ham (Not Spam)** using **Logistic Regression**. The model is trained on a labeled dataset and utilizes **TF-IDF Vectorization** for text preprocessing.

## Features
- **Text Preprocessing**: Converts email messages into numerical features using **TF-IDF Vectorization**.
- **Label Encoding**: Assigns numerical labels (1 for Ham, 0 for Spam).
- **Machine Learning Model**: Uses **Logistic Regression** for classification.
- **Model Evaluation**: Computes accuracy scores for training and test datasets.
- **Real-Time Predictions**: Allows users to input an email message and classify it as spam or ham.

## Data Requirements
The system requires a dataset (`mail_data.csv`) with the following columns:
- `Category`: Label (Spam or Ham)
- `Message`: Email content

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-mail-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd spam-mail-prediction
   ```
3. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Usage
1. Ensure that `mail_data.csv` is in the project directory.
2. Run the script:
   ```bash
   python spam_mail_prediction.py
   ```
3. The script will preprocess the data, train the model, and compute accuracy scores.
4. Enter an email message for prediction, and the system will classify it as **Spam or Ham**.

## Workflow & Key Functions
1. **Data Loading & Preprocessing**
   - Reads `mail_data.csv` into a Pandas DataFrame.
   - Fills missing values with empty strings.
   - Encodes labels (`Spam = 0`, `Ham = 1`).

2. **Text Vectorization**
   - Converts text into numerical feature vectors using **TF-IDF Vectorization**.
   - Example:
     ```python
     feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
     X_train_features = feature_extraction.fit_transform(X_train)
     ```

3. **Model Training & Evaluation**
   - Splits data into training and testing sets (`train_test_split`).
   - Trains a **Logistic Regression** model.
   - Computes accuracy scores for both training and test datasets.
   - Example:
     ```python
     model = LogisticRegression()
     model.fit(X_train_features, Y_train)
     print("Accuracy:", accuracy_score(Y_test, model.predict(X_test_features)))
     ```

4. **Real-Time Spam Detection**
   - Accepts user input and classifies emails as Spam or Ham.
   - Example:
     ```python
     input_mail = ["Win a free iPhone! Click here to claim your prize!"]
     input_data_features = feature_extraction.transform(input_mail)
     prediction = model.predict(input_data_features)
     ```

## Example Output
```
Accuracy on Training Data: 98%
Accuracy on Test Data: 96%
Enter an email message: "Congratulations! You've won a free vacation. Click to claim."
Prediction: Spam Mail
```

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn`
