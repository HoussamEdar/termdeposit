Project: Marketing Analysis


The main issue we're addressing is that numerous banks are interested in identifying the customer 
segments most likely to take up an offer to better target their efforts effectively
and efficiently invest their resources where it matters most.
To start off our analysis process.
Gain insights into the problem at hand more clearly and thoroughly 
understand the data set provided by running various classifiers and exploring different strategies to enhance their performance, 
for predicting campaign responses. The original dataset consists of the banking dataset comprising 41188 rows and 21 characteristics/features
in total most of which were categorical in nature prompting us to convert the data and introduce variables to delve deeper into each categorical variable present, 
within it. The specific categorical variables included 'job' 'marital' 'education' 'default' 'housing' 'loan' 'contact' 'month' 'day_of_week' and 'poutcome'.

Our main research question is:

To predict whether a customer avails term deposit or not.

This  data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution.
The classification goal is to predict if the client will subscribe a term deposit (variable y)


Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")
  <img width="437" alt="image" src="https://github.com/user-attachments/assets/ecd69a9e-3818-437c-a6ab-7d21aa926c91">
  <img width="512" alt="image" src="https://github.com/user-attachments/assets/6a67c462-902d-4273-897a-3d1b1e53fe2c">
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

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>


