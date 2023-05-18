
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

![Money Losses and Saved](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi6FyzwzRO0ZJdvJJLj9_m1vqiHS4NdsncN2InC2QQ2-KX0RHDZ13IqxLDx-vZJJoGfqF3BnqIj2bcBfPwgopEAY-mBvoYmwxfPzF4ZntGAKy4uHAy8ZNlQoMpyBMKdpE6Ux4ipW2v4XDkuveoKDeIU9aRvXLhcP457u_4UclB6axfEER7PfwPKAX-Y/s1600/money-losses-and-saved.png)
## Conclusions

* There are a lot of missing values in the application_train.csv (9,152,465 items) and application_test.csv (1,404,419) but it's already handled using mean technique for numerical variables and mode technique for categorical variables
* There is no duplicated values in the dataset
 * I've created new features such as age, annuity income ratio, and loan duration.
* Outliers has been handled by deleting their non-sense values
* Target variables consists of 91.3% Non-Defaulter (Accepted) and 8.7% Defaulter (Rejected)
* Weight of Evidence (WOE) and Information Value (IV) has been applied in feature selection
* Machine learning model using Logistic regression resulting Mean AUROC about 73.5%, Gini around 47.0%, and AUCPR approximately 21.8%
* If the result is True Positive (If my machine predicts that the applicant will default, and they actually do default), the company will save approximately 5,000,000,000 IDR. Otherwise, if the result is False Negative (If my machine predicts that the applicant will not default, but they actually do default), the company will lose approximately 20,000,000,000 IDR.
* These variables (The percentage of True Positive/Negative and False Positive/Negative) depend on the quality of metrics from my machine learning model described above. 
* These low metrics are due to the lack of information value (IV) between features. Moreover, there are many CSV files that contain features with higher potential IV that I couldn't merge into application_train.csv and application_test.csv. These files include:
	* bureau.csv
	* bureau_balance.csv
	* credit_card_balance.csv
	* installments_payments.csv
	* POS_CASH_balance.csv
	* previous_application.csv
* I hope in the future I can purchase a more advanced laptop/computer to merge these files, as my current laptop (4GB RAM) crashes several times when attempting to do so.
