# Marketing Analysis: Predicting Term Deposit Subscriptions

## Project Overview

This project focuses on identifying customer segments most likely to subscribe to a term deposit at a Portuguese banking institution. By predicting customer behavior, banks can better target their marketing efforts, allocating resources efficiently and effectively where they are most likely to yield results.

The dataset provided consists of 41,188 records with 21 features, many of which are categorical. Our analysis involves converting these categorical variables into a suitable format and introducing additional variables to deepen our understanding of the data. The main goal is to predict whether a client will subscribe to a term deposit, based on various demographic and interaction-related factors.

## Research Question

**Can we accurately predict whether a customer will subscribe to a term deposit based on their profile and past interactions?**

## Dataset Description

The dataset relates to direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. The classification goal is to predict whether a client will subscribe to a term deposit (variable `y`).



## Analysis Process

1. **Data Preprocessing**: Converting categorical variables into numerical formats, handling missing values, and feature engineering to enhance model performance.
2. **Exploratory Data Analysis (EDA)**: Understanding the distribution and relationships between variables to gain insights.
3. **Model Training**: Running various classifiers and exploring different strategies to enhance their performance in predicting campaign responses.
4. **Model Evaluation**: Assessing the performance of each model using appropriate metrics to ensure accuracy and reliability.

## Conclusion

The aim of this project is to develop a reliable predictive model that can help the bank target the right customers for term deposit offers. By understanding the key factors that influence a customer's decision, the bank can optimize its marketing strategy, resulting in better outcomes.

## Contact

For any questions or further information, please contact [Your Name] at [Your Email].



# Term Deposit Subscription Prediction - Model Training

This repository contains scripts and instructions for training machine learning models to predict whether a client will subscribe to a term deposit. The models are trained using a dataset from a Portuguese banking institution, with features such as client demographics, previous interactions, and financial details.

## Project Overview

The goal of this project is to develop a predictive model that can accurately forecast client subscription to a term deposit based on available features. Various models, including Random Forest, Naive Bayes, SVM, and K-Nearest Neighbors, have been trained and evaluated.

## Dataset

The dataset used for training contains the following features:

- **Age**: Age of the client.
- **Job**: Type of job.
- **Marital**: Marital status.
- **Education**: Level of education.
- **Default**: Whether the client has credit in default.
- **Housing**: Whether the client has a housing loan.
- **Loan**: Whether the client has a personal loan.
- **Contact**: Preferred contact communication type.
- **Month**: Last contact month of the year.
- **Day**: Last contact day of the month.
- **Duration**: Duration of the last call in seconds.
- **Campaign**: Number of contacts performed during this campaign.
- **Pdays**: Number of days since the client was last contacted from a previous campaign.
- **Previous**: Number of contacts performed before this campaign.
- **Poutcome**: Outcome of the previous marketing campaign.

## Model Training

### 1. Data Preprocessing

The preprocessing steps include:

- **Handling Missing Values**: Impute or remove missing data as necessary.
- **Categorical Encoding**: Convert categorical features into numerical values using one-hot encoding.
- **Feature Scaling**: Normalize or standardize the features to improve model performance.
- **Train-Test Split**: Split the data into training and testing sets.

### 2. Model Selection

The following models were trained and evaluated:

- **Naive Bayes**
- **Random Forest**
- **SVM (Support Vector Machine)**
- **K-Nearest Neighbors (KNN)**

### 3. Model Evaluation

Each model was evaluated using the following metrics:

- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **ROC AUC**: The area under the receiver operating characteristic curve.

### 4. Random Forest and RF2 Comparison

- **Random Forest**: Standard Random Forest model with hyperparameter tuning.
- **RF2**: A variant of Random Forest with specific tweaks to improve recall.

### 5. Hyperparameter Tuning

Grid Search was used to optimize the hyperparameters for the Random Forest model, comparing it to its variant (RF2). Key hyperparameters include:

- **n_estimators**: Number of trees in the forest.
- **max_depth**: Maximum depth of the trees.
- **min_samples_split**: Minimum number of samples required to split an internal node.
- **min_samples_leaf**: Minimum number of samples required to be at a leaf node.

## Model Training Scripts

### `train_model.py`

This script performs the following tasks:

1. **Load Data**: Reads the dataset from a CSV file.
2. **Preprocessing**: Applies the necessary preprocessing steps.
3. **Train Models**: Trains multiple models and saves them as pickle files.
4. **Evaluation**: Evaluates the models and prints the performance metrics.

### `grid_search_rf.py`

This script is dedicated to:

1. **Grid Search**: Performs hyperparameter tuning using GridSearchCV for the Random Forest model.
2. **Model Comparison**: Compares the tuned Random Forest model with its variant (RF2).
3. **Save Best Model**: Saves the best-performing model as `rf.pkl`.

## Usage

### Prerequisites

- Python 3.8+
- pandas
- scikit-learn
- joblib


# Term Deposit Subscription Prediction App

This Streamlit web application allows users to predict whether a client will subscribe to a term deposit based on their profile. The app supports both single client predictions and batch predictions using a pre-trained Random Forest model.

  <img width="437" alt="image" src="https://github.com/user-attachments/assets/ecd69a9e-3818-437c-a6ab-7d21aa926c91">
  <img width="512" alt="image" src="https://github.com/user-attachments/assets/6a67c462-902d-4273-897a-3d1b1e53fe2c">
  
## Features

### 1. Single Prediction
- **Input Fields**: Users can input client details such as age, job, marital status, education, and more.
- **Prediction**: The app uses a trained Random Forest model to predict whether the client will subscribe to a term deposit.
- **Result Display**: The prediction result is displayed immediately, indicating whether the client is likely to subscribe or not.

### 2. Batch Prediction
- **CSV Upload**: Users can upload a CSV file containing multiple client profiles.
- **Batch Processing**: The app processes the CSV file, performs predictions for all clients, and displays the results.
- **Downloadable Results**: Users can download the prediction results as a CSV file.
## How to Use

### Single Prediction
1. Select the **Single Prediction** page from the sidebar.
2. Enter the client details in the provided fields.
3. Click on the **Predict** button to see whether the client is likely to subscribe to a term deposit.

### Batch Prediction
1. Select the **Batch Prediction** page from the sidebar.
2. Upload a CSV file containing the client data.
3. The app will process the file and display the prediction results.
4. Click the **Download Predictions** button to download the results as a CSV file.
### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HoussamEdar/termdeposit


