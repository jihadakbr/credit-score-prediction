
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
* Target variable = 0 → Rejected for a loan → Defaulter
* Target variable = 1 → Accepted for a loan → Non-Defaulter
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

![scorecard development](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjHB9YwwfhZiD5MaOhmsMkrzamGuQv6XVZt3vz-kd5ERJtHLZoyIGpIRKPfuR0fxkEbFbTc-Ynixia-8ZU-Gzc2pbeCCtHI3NPtbN_KLW2pl9cO1GEzZExwleKe7MNSFFNrIvPo3TbAqfHHf27VVUCdBNaGLWGinq98FsoxYwtR0vqcWA6mkRRiqXdL/s1600/scorecard-development.png)

![Money Losses and Saved](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgIisG2vEjrxyz1yjQ0h7uetpuneW2vJvdpBzDgNuc-vzlgr9KLVnwExDnfr0vNh_ZJs0mEq0ThjaYOgw-lXylETXsyFiVCW-VaxOZv7XYa9a2XFo2Mw-SbsP3rgYlgX9DAUyffLcZWd49lrV_oI6VdgduhEfTntzmsagDmxrCjosMYLacBQw5owjNj/s1600/money-losses-and-saved.png)
## Conclusions

* The application_train.csv (307511 rows and 122 columns) and application_test.csv (48744 rows and 121 columns) contain numerous missing values and outliers, which have been handled using the WOE binning technique.
* No duplicate values are present in the dataset.
* Additional features, including age, annuity income ratio, and loan duration, have been created.
* The target variables consist of 91.9% non-defaulters (accepted) and 8.1% defaulters (rejected).
* Feature selection has been performed using Weight of Evidence (WOE) and Information Value (IV).
* Logistic regression was employed in a machine learning model, yielding the following metrics: threshold ≈ 0.23, accuracy ≈ 0.90, precision ≈ 0.93, recall ≈ 0.96, F1 ≈ 0.94, AUROC ≈ 0.74, Gini ≈ 0.48, and AUCPR ≈ 0.97. These metrics exhibit strong performance in credit risk modeling.
* Consequently, the company is expected to save around 30,000,000,000 IDR while incurring a loss of approximately 100,000,000 IDR.
* The high or low percentages of True Positive/Negative and False Positive/Negative depend on the metrics of the machine learning model mentioned above.
* Furthermore, we can enhance them further by incorporating features with higher information value (IV). Several CSV files encompassing such features possess significant IV potential, yet I was unable to merge them into the application_train.csv and application_test.csv datasets. These files comprise:
    * bureau.csv
    * bureau_balance.csv 
    * credit_card_balance.csv 
    * installments_payments.csv 
    * POS_CASH_balance.csv 
    * previous_application.csv 

   that contain features with higher potential IV but couldn't be merged into application_train.csv and application_test.csv. This limitation is due to the current laptop (4GB RAM) experiencing crashes when attempting to merge these files.
* It is hoped that in the future, a more advanced laptop/computer can be acquired to successfully merge these files.
