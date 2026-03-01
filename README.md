# Identifying Fraudulent Transactions in Financial Payment Service Data using Ensemble Learning

This project utilizes artificial payments data to develop a classifier for identifying fraudulent transactions. The data consists of 6,362,620 rows and 11 columns, and is available in CSV format. 

The data can be accessed at: https://www.kaggle.com/datasets/arunavakrchakraborty/financial-payment-services-fraud-data

**Handling Imbalanced Data:**

The workflow details how we handle the highly imbalanced dataset where fraud cases are less than 1% of the total data. The dataset is processed using three different strategies to balance fraud and non-fraud cases before training machine learning models:

|1. Undersampling Non-Fraud|2. Oversampling using SMOTE|3. Balanced Class Weight|
|:-|:-|:-|
|The non-fraudulent cases are reduced to increase the fraud rate to approximately 33%.|Synthetic Minority Over-sampling Technique (SMOTE) is used to increase fraud cases, raising the fraud rate to about 50%.|The entire dataset is used without resampling, but class weights are adjusted to balance fraud cases during model training.|

**Model Development:**

For each type of sampling, we implement the following Supervised Machine Learning methods for classification:

1. **Logistic Regression (Baseline)** 
2. **Ensemble Classifier:** Random Forest, Gradient Boosting, Bagging, XGBoost
3. **Anomaly Detection:** Autoencoder, One-Class Support Vector Machine

Each model is evaluated using **Area Under the Curve - Precision Recall (AUC-PR)** on the Training, Testing, and Validation Datasets:

* AUC-PR focuses only on Precision (how many predicted frauds are actually frauds) and Recall (how many actual frauds were correctly detected).
* The Precision-Recall curve better captures performance when false positives and false negatives matter more than true negatives.

## Key Takeaways

* Our best-performing approach—**oversampling with SMOTE and modeling with XGBoost**—helped balance recall and precision, making it the most effective method for identifying fraudulent transactions.
  * **Precision: 0.9875, Recall: 0.9996, AUC-PR: 0.9999**</mark>
  * However, improving model performance further will require incorporating richer features, such as merchant identities and geographic transaction patterns, to enhance predictive power and reduce false positives.
    
* Looking ahead, a key improvement would be leveraging distributed computing frameworks, such as Apache Spark, to scale fraud detection modeling efficiently. Given the vast volume of financial transactions in real-world applications, single-machine models are not practical for handling large datasets or complex hyperparameter tuning. Distributed systems enable parallel processing, allowing us to train more sophisticated models faster and at scale. Implementing such solutions would enhance fraud detection accuracy and ensure models remain robust in production environments.



