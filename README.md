
# Credit Score Prediction - Home Credit Indonesia

Home Credit Indonesia is currently using various statistical methods and Machine Learning to make credit score predictions. Now, we ask you to unlock the maximum potential of our data. By doing so, we can ensure that:
* Customers who are capable of repayment are not rejected when applying for a loan.
* Loans can be given with a principal, maturity, and repayment calendar that will motivate customers to succeed.
Evaluation will be done by checking how deep your understanding of the analysis is. Note that you need to use at least Logistic Regression to construct your machine learning models. After that, create a presentation slide containing end-to-end modeling analysis results along with business recommendations (maximum 10 pages).


## Dataset Source

Home Credit Indonesia's internal database
## Dataset Files

1. application_test.csv
2. application_train.csv
3. bureau.csv
4. bureau_balance.csv
5. credit_card_balance.csv
6. HomeCredit_columns_description.csv
7. installments_payments.csv
8. POS_CASH_balance.csv
9. previous_application.csv
10. sample_submission.csv

Target Variable Description:
* Target variable = 1 → Rejected for a loan → Defaulter
* Target variable = 0 → Accepted for a loan → Non-Defaulter
## Tools

* Programming language: Python.
* Data Tool: Jupyter Notebook.
* Reporting Tool: Microsoft PowerPoint.
## The Project Workflow

1. Problem Formulation    
2. Data Collecting
3. Data Understanding
4. Data preprocessing
5. Exploratory Data Analysis (EDA) and Data Visualization
7. Model Selection and Building
8. Scorecard Development

## Results

![scorecard development](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgh9dmvLMyTnrbrhulhCqjd5jm6wVXoKXGNywt47z-4cAiybgJNYTjoEGUdSqZVe-tgvbpXOfUxjzYVtFCul1ShcbEW_NhNmvNFudSV7DX-BSBYgdsaREbOrkxzglExBJcMLEXgCrmd6Pfyp8apIjqp0dCxluWfnM8hve9Npm5Lyzw1dnlLdHT5X3nP/s1600/scorecard-development.png)

![Money Losses and Saved](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjRuomqTI-ORGJeL1QCCpfMR5yVY0uILPpmmsW7ZMdwaLw4EFfZrJbDMj5UshVhefHjCV6XpARjBDcDhIt3jtjr4UygoXOX1c68eTzzG99hLCJcXmuNOMf6RwsDEn8maQ9vroWv993KEZ3jVt2Ndw0Ja9Gxwp58jA5Lycn5TtLMNIxLsQYTXXX5aok0/s1600/money-losses-and-saved.png)
## Conclusions

* The application_train.csv (9,152,465 items) and application_test.csv (1,404,419) contain numerous missing values, which have been addressed by applying the mean technique for numerical variables and the mode technique for categorical variables.
* No duplicate values are present in the dataset.
* Additional features, including age, annuity income ratio, and loan duration, have been created.
* Outliers have been handled by removing their nonsensical values.
* The target variables consist of 91.3% non-defaulters (accepted) and 8.7% defaulters (rejected).
* Feature selection has been performed using Weight of Evidence (WOE) and Information Value (IV).
* Logistic regression was employed in a machine learning model, yielding the following metrics: threshold ≈ 0.23, accuracy ≈ 0.23, precision ≈ 0.10, recall ≈ 97.0, F1 ≈ 0.18, AUROC ≈ 0.73, Gini ≈ 0.47, and AUCPR ≈ 0.22.
* Consequently, the company is expected to save around 400,000,000 IDR while incurring a loss of approximately 50,000,000 IDR.
* The high or low percentages of True Positive/Negative and False Positive/Negative depend on the metrics of the machine learning model mentioned above.
* The lower metrics can be attributed to the lack of Information Value (IV) between features. Additionally, there are several CSV files, such as 
    * bureau.csv
    * bureau_balance.csv 
    * credit_card_balance.csv 
    * installments_payments.csv 
    * POS_CASH_balance.csv 
    * previous_application.csv 

   that contain features with higher potential IV but couldn't be merged into application_train.csv and application_test.csv. This limitation is due to the current laptop (4GB RAM) experiencing crashes when attempting to merge these files.
* It is hoped that in the future, a more advanced laptop/computer can be acquired to successfully merge these files.
