## warning
import warnings
warnings.filterwarnings("ignore")

## random state
random_state_ = 42

## for data
import numpy as np
import pandas as pd
import math

## for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

## for statistical tests
from scipy import stats

## for machine learning
from sklearn import model_selection, pipeline, feature_selection, metrics, linear_model, base

#########################################################################################################################
#########################################################################################################################

def column_check(df):
    unique_values = df.nunique()
    dtypes = df.dtypes
    examples = df.apply(lambda x: ", ".join(x.sample(n=5, random_state=random_state_).astype(str)))

    info = pd.DataFrame({"Unique Values": unique_values, "Data Types": dtypes, "Examples": examples})
    return info

def object_input_check(df, columns):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        print(f"Processing column: {col}")
        rows_with_number = []

        if col in df.columns:
            for index, value in enumerate(df[col]):
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    rows_with_number.append((index, str(value)))
            if rows_with_number:
                print("Number found in the 'object' column.")
                print("Rows with number values: (index from 0)")
                for row in rows_with_number:
                    print(f"Row {row[0]}: {row[1]}")
            else:
                print("No number found in the 'object' column.")
        else:
            print(f"Column '{col}' not found in the DataFrame.")
        print("")

def count_outliers(data, columns):
    outlier_percentage = {}
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        upper_limit = q3 + 1.5 * (q3-q1)
        lower_limit = q1 - 1.5 * (q3-q1)

        col_outliers = (data[col] < lower_limit) | (data[col] > upper_limit)
        outlier_percentage[col] = f"{col_outliers.mean() * 100:.2f}%"

    return outlier_percentage

def plot_histograms(df, selected_cols):
    num_rows = math.ceil(len(selected_cols) / 4)
    fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 4*num_rows))
    axs = axs.ravel()
    for i, col in enumerate(selected_cols):
        axs[i].hist(df[col], bins=50)
        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].grid()

    for i in range(len(selected_cols), len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.show()
    
def cat_dist(data, col_x, col_target, target_0_label, target_1_label, colour_number, label_x_offset=0.05, legend_loc="upper right", x_label_rotation=0):
    palette = ["husl", "hls", "Spectral", "coolwarm", "viridis"]
    sns.set_style("whitegrid")
    
    sorted_data = data[col_x].value_counts().sort_values(ascending=False).index
    hue_order = sorted(data[col_target].unique(), reverse=True)
    order = sorted_data.tolist()
    
    ax = sns.countplot(data=data, x=col_x, hue=col_target, order=order, hue_order=hue_order, palette=palette[colour_number])
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_1_label, target_0_label], loc=legend_loc)
    plt.grid(False)
    plt.xticks(rotation=x_label_rotation)
    
    for i in ax.patches:
        ax.text(i.get_x()+label_x_offset, i.get_height()+1000, int(i.get_height()), fontsize=11)

def cat_dist_top5(data, col_x, col_target, target_0_label, target_1_label, colour_number, label_x_offset=0.05, legend_loc="upper right", x_label_rotation=0):
    palette = ["husl","hls","Spectral","coolwarm", "viridis"]
    sns.set_style("whitegrid")
    
    sorted_data = data[col_x].value_counts().sort_values(ascending=False).index
    hue_order = sorted(data[col_target].unique(), reverse=True)
    order = sorted_data.tolist()
    
    top5 = data[col_x].value_counts().index[:5]
    ax = sns.countplot(data=data[data[col_x].isin(top5)], x=col_x, hue=col_target, order=top5, hue_order=hue_order,
                       palette=palette[colour_number])
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_1_label, target_0_label], loc=legend_loc)
    plt.grid(False)
    plt.title(f"Top 5 {col_x} categories", fontsize=18, pad=20)
    plt.xticks(rotation=x_label_rotation)

    for i in ax.patches:
        ax.text(i.get_x()+label_x_offset, i.get_height()+300, int(i.get_height()), fontsize=11)
    
def target_dist(data, col_target, target_0_label, target_1_label):
    mpl.rcParams["font.size"] = 11
    r = data.groupby(col_target)[col_target].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[target_0_label, target_1_label], radius=1.5, autopct="%1.1f%%", shadow=True, startangle=45,
           colors=["#ff9999", "#66b3ff"])
    ax.set_aspect("equal")
    ax.set_frame_on(False)      

def corrr(data):
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(10,10))
    sns.heatmap(data, annot=True, mask=mask, cmap='coolwarm', annot_kws={"size": 7})
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Correlation Matrix", fontsize=15, fontweight='bold')
    plt.show()    

def col_to_drop(df, columns_list):
    df.drop(columns = columns_list, inplace = True)    

def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df
    
def woe_discrete(df, cat_variabe_name, y_df):
    df = pd.concat([df[cat_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

def woe_ordered_continuous(df, continuous_variabe_name, y_df):
    df = pd.concat([df[continuous_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    sns.set()
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)  

ref_categories = ["NAME_CONTRACT_TYPE:Cash loans","CODE_GENDER:F",
                  "EXT_SOURCE_2:>0.855","EXT_SOURCE_3:>0.896","EXT_SOURCE_1:>0.952","AGE_IN_YEARS:>69","REGION_RATING_CLIENT:1",
                  "DAYS_LAST_PHONE_CHANGE:>0.0","DAYS_ID_PUBLISH:>0.0","DAYS_EMPLOYED:>365243.0","FLAG_DOCUMENT_3:1",
                  "REG_CITY_NOT_LIVE_CITY:0","DAYS_REGISTRATION:>0.0","AMT_GOODS_PRICE:>2254500.0","REGION_POPULATION_RELATIVE:>0.0725",
                  "LOAN_DURATION:>39.702"]
                  
class WoE_Binning(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, X):
        self.X = X
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_new = X.loc[:, "NAME_CONTRACT_TYPE:Cash loans":"NAME_CONTRACT_TYPE:Revolving loans"]
        X_new = pd.concat([X_new, X.loc[:, "CODE_GENDER:F":"CODE_GENDER:M"]], axis = 1)
                
        X_new["EXT_SOURCE_2:missing"] = np.where(X["EXT_SOURCE_2"].isnull(), 1, 0)    
        X_new["EXT_SOURCE_2:<=0.171"] = np.where((X["EXT_SOURCE_2"] <= 0.171), 1, 0)
        X_new["EXT_SOURCE_2:0.171–0.285"] = np.where((X["EXT_SOURCE_2"] > 0.171) & (X["EXT_SOURCE_2"] <= 0.285), 1, 0)
        X_new["EXT_SOURCE_2:0.285–0.399"] = np.where((X["EXT_SOURCE_2"] > 0.285) & (X["EXT_SOURCE_2"] <= 0.399), 1, 0)
        X_new["EXT_SOURCE_2:0.399–0.513"] = np.where((X["EXT_SOURCE_2"] > 0.399) & (X["EXT_SOURCE_2"] <= 0.513), 1, 0)
        X_new["EXT_SOURCE_2:0.513–0.684"] = np.where((X["EXT_SOURCE_2"] > 0.513) & (X["EXT_SOURCE_2"] <= 0.684), 1, 0)
        X_new["EXT_SOURCE_2:0.684–0.741"] = np.where((X["EXT_SOURCE_2"] > 0.684) & (X["EXT_SOURCE_2"] <= 0.741), 1, 0)
        X_new["EXT_SOURCE_2:0.741–0.855"] = np.where((X["EXT_SOURCE_2"] > 0.741) & (X["EXT_SOURCE_2"] <= 0.855), 1, 0)
        X_new["EXT_SOURCE_2:>0.855"] = np.where((X["EXT_SOURCE_2"] > 0.855), 1, 0)
        
        X_new["EXT_SOURCE_3:missing"] = np.where(X["EXT_SOURCE_3"].isnull(), 1, 0)
        X_new["EXT_SOURCE_3:<=0.18"] = np.where((X["EXT_SOURCE_3"] <= 0.18), 1, 0)
        X_new["EXT_SOURCE_3:0.18–0.299"] = np.where((X["EXT_SOURCE_3"] > 0.18) & (X["EXT_SOURCE_3"] <= 0.299), 1, 0)
        X_new["EXT_SOURCE_3:0.299–0.478"] = np.where((X["EXT_SOURCE_3"] > 0.299) & (X["EXT_SOURCE_3"] <= 0.478), 1, 0)
        X_new["EXT_SOURCE_3:0.478–0.598"] = np.where((X["EXT_SOURCE_3"] > 0.478) & (X["EXT_SOURCE_3"] <= 0.598), 1, 0)
        X_new["EXT_SOURCE_3:0.598–0.717"] = np.where((X["EXT_SOURCE_3"] > 0.598) & (X["EXT_SOURCE_3"] <= 0.717), 1, 0)
        X_new["EXT_SOURCE_3:0.717–0.896"] = np.where((X["EXT_SOURCE_3"] > 0.717) & (X["EXT_SOURCE_3"] <= 0.896), 1, 0)
        X_new["EXT_SOURCE_3:>0.896"] = np.where((X["EXT_SOURCE_3"] > 0.896), 1, 0)

        X_new["EXT_SOURCE_1:missing"] = np.where(X["EXT_SOURCE_1"].isnull(), 1, 0)
        X_new["EXT_SOURCE_1:<=0.202"] = np.where((X["EXT_SOURCE_1"] <= 0.202), 1, 0)
        X_new["EXT_SOURCE_1:0.202–0.327"] = np.where((X["EXT_SOURCE_1"] > 0.202) & (X["EXT_SOURCE_1"] <= 0.327), 1, 0)
        X_new["EXT_SOURCE_1:0.327–0.452"] = np.where((X["EXT_SOURCE_1"] > 0.327) & (X["EXT_SOURCE_1"] <= 0.452), 1, 0)
        X_new["EXT_SOURCE_1:0.452–0.639"] = np.where((X["EXT_SOURCE_1"] > 0.452) & (X["EXT_SOURCE_1"] <= 0.639), 1, 0)
        X_new["EXT_SOURCE_1:0.639–0.764"] = np.where((X["EXT_SOURCE_1"] > 0.639) & (X["EXT_SOURCE_1"] <= 0.764), 1, 0)
        X_new["EXT_SOURCE_1:0.764–0.952"] = np.where((X["EXT_SOURCE_1"] > 0.764) & (X["EXT_SOURCE_1"] <= 0.952), 1, 0)
        X_new["EXT_SOURCE_1:>0.952"] = np.where((X["EXT_SOURCE_1"] > 0.952), 1, 0)

        X_new["AGE_IN_YEARS:<=28"] = np.where((X["AGE_IN_YEARS"] <= 28), 1, 0)
        X_new["AGE_IN_YEARS:28–32"] = np.where((X["AGE_IN_YEARS"] > 28) & (X["AGE_IN_YEARS"] <= 32), 1, 0)
        X_new["AGE_IN_YEARS:32–36"] = np.where((X["AGE_IN_YEARS"] > 32) & (X["AGE_IN_YEARS"] <= 36), 1, 0)
        X_new["AGE_IN_YEARS:36–40"] = np.where((X["AGE_IN_YEARS"] > 36) & (X["AGE_IN_YEARS"] <= 40), 1, 0)
        X_new["AGE_IN_YEARS:40–44"] = np.where((X["AGE_IN_YEARS"] > 40) & (X["AGE_IN_YEARS"] <= 44), 1, 0)
        X_new["AGE_IN_YEARS:44–50"] = np.where((X["AGE_IN_YEARS"] > 44) & (X["AGE_IN_YEARS"] <= 50), 1, 0)
        X_new["AGE_IN_YEARS:50–56"] = np.where((X["AGE_IN_YEARS"] > 50) & (X["AGE_IN_YEARS"] <= 56), 1, 0)
        X_new["AGE_IN_YEARS:56–62"] = np.where((X["AGE_IN_YEARS"] > 56) & (X["AGE_IN_YEARS"] <= 62), 1, 0)
        X_new["AGE_IN_YEARS:62–69"] = np.where((X["AGE_IN_YEARS"] > 62) & (X["AGE_IN_YEARS"] <= 69), 1, 0)
        X_new["AGE_IN_YEARS:>69"] = np.where((X["AGE_IN_YEARS"] > 69), 1, 0)
        
        X_new["REGION_RATING_CLIENT:1"] = np.where((X["REGION_RATING_CLIENT"] == 1), 1, 0)
        X_new["REGION_RATING_CLIENT:2"] = np.where((X["REGION_RATING_CLIENT"] == 2), 1, 0)
        X_new["REGION_RATING_CLIENT:3"] = np.where((X["REGION_RATING_CLIENT"] == 3), 1, 0)
        
        X_new["DAYS_LAST_PHONE_CHANGE:missing"] = np.where(X["DAYS_LAST_PHONE_CHANGE"].isnull(), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:<=-2289.067"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] <= -2289.067), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-2289.067–(-1716.8)"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -2289.067) & (X["DAYS_LAST_PHONE_CHANGE"] <= -1716.8), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-1716.8–(-1144.533)"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -1716.8) & (X["DAYS_LAST_PHONE_CHANGE"] <= -1144.533), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-1144.533–(-572.267)"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -1144.533) & (X["DAYS_LAST_PHONE_CHANGE"] <= -572.267), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-572.267–0.0"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -572.267) & (X["DAYS_LAST_PHONE_CHANGE"] <= 0.0), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:>0.0"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > 0.0), 1, 0)

        X_new["DAYS_ID_PUBLISH:<=-4680.867"] = np.where((X["DAYS_ID_PUBLISH"] <= -4680.867), 1, 0)
        X_new["DAYS_ID_PUBLISH:-4680.867–(-4042.567)"] = np.where((X["DAYS_ID_PUBLISH"] > -4680.867) & (X["DAYS_ID_PUBLISH"] <= -4042.567), 1, 0)
        X_new["DAYS_ID_PUBLISH:-4042.567–(-3617.033)"] = np.where((X["DAYS_ID_PUBLISH"] > -4042.567) & (X["DAYS_ID_PUBLISH"] <= -3617.033), 1, 0)
        X_new["DAYS_ID_PUBLISH:-3617.033–(-3191.5)"] = np.where((X["DAYS_ID_PUBLISH"] > -3617.033) & (X["DAYS_ID_PUBLISH"] <= -3191.5), 1, 0)
        X_new["DAYS_ID_PUBLISH:-3191.5–(-2553.2)"] = np.where((X["DAYS_ID_PUBLISH"] > -3191.5) & (X["DAYS_ID_PUBLISH"] <= -2553.2), 1, 0)
        X_new["DAYS_ID_PUBLISH:-2553.2–(-1702.133)"] = np.where((X["DAYS_ID_PUBLISH"] > -2553.2) & (X["DAYS_ID_PUBLISH"] <= -1702.133), 1, 0)
        X_new["DAYS_ID_PUBLISH:-1702.133–(-851.067)"] = np.where((X["DAYS_ID_PUBLISH"] > -1702.133) & (X["DAYS_ID_PUBLISH"] <= -851.067), 1, 0)
        X_new["DAYS_ID_PUBLISH:-851.067–0.0"] = np.where((X["DAYS_ID_PUBLISH"] > -851.067) & (X["DAYS_ID_PUBLISH"] <= 0.0), 1, 0)
        X_new["DAYS_ID_PUBLISH:>0.0"] = np.where((X["DAYS_ID_PUBLISH"] > 0.0), 1, 0)

        X_new["DAYS_EMPLOYED:<=-5140.167"] = np.where((X["DAYS_EMPLOYED"] <= -5140.167), 1, 0)
        X_new["DAYS_EMPLOYED:-5140.167–7631.667"] = np.where((X["DAYS_EMPLOYED"] > -5140.167) & (X["DAYS_EMPLOYED"] <= 7631.667), 1, 0)
        X_new["DAYS_EMPLOYED:7631.667–365243.0"] = np.where((X["DAYS_EMPLOYED"] > 7631.667) & (X["DAYS_EMPLOYED"] <= 365243.0), 1, 0)
        X_new["DAYS_EMPLOYED:>365243.0"] = np.where((X["DAYS_EMPLOYED"] > 365243.0), 1, 0)

        X_new["FLAG_DOCUMENT_3:0"] = np.where((X["FLAG_DOCUMENT_3"] == 0), 1, 0)
        X_new["FLAG_DOCUMENT_3:1"] = np.where((X["FLAG_DOCUMENT_3"] == 1), 1, 0)

        X_new["REG_CITY_NOT_LIVE_CITY:0"] = np.where((X["REG_CITY_NOT_LIVE_CITY"] == 0), 1, 0)
        X_new["REG_CITY_NOT_LIVE_CITY:1"] = np.where((X["REG_CITY_NOT_LIVE_CITY"] == 1), 1, 0)
        
        X_new["DAYS_REGISTRATION:<=-10380.967"] = np.where((X["DAYS_REGISTRATION"] <= -10380.967), 1, 0)
        X_new["DAYS_REGISTRATION:-10380.967–(-8195.5)"] = np.where((X["DAYS_REGISTRATION"] > -10380.967) & (X["DAYS_REGISTRATION"] <= -8195.5), 1, 0)
        X_new["DAYS_REGISTRATION:-8195.5–(-5463.667)"] = np.where((X["DAYS_REGISTRATION"] > -8195.5) & (X["DAYS_REGISTRATION"] <= -5463.667), 1, 0)
        X_new["DAYS_REGISTRATION:-5463.667–(-3278.2)"] = np.where((X["DAYS_REGISTRATION"] > -5463.667) & (X["DAYS_REGISTRATION"] <= -3278.2), 1, 0)
        X_new["DAYS_REGISTRATION:-3278.2–(-1639.1)"] = np.where((X["DAYS_REGISTRATION"] > -3278.2) & (X["DAYS_REGISTRATION"] <= -1639.1), 1, 0)
        X_new["DAYS_REGISTRATION:-1639.1–0.0"] = np.where((X["DAYS_REGISTRATION"] > -1639.1) & (X["DAYS_REGISTRATION"] <= 0.0), 1, 0)
        X_new["DAYS_REGISTRATION:>0.0"] = np.where((X["DAYS_REGISTRATION"] > 0.0), 1, 0)

        X_new["AMT_GOODS_PRICE:missing"] = np.where(X["AMT_GOODS_PRICE"].isnull(), 1, 0)
        X_new["AMT_GOODS_PRICE:<=151200.0"] = np.where((X["AMT_GOODS_PRICE"] <= 151200.0), 1, 0)
        X_new["AMT_GOODS_PRICE:151200.0–261900.0"] = np.where((X["AMT_GOODS_PRICE"] > 151200.0) & (X["AMT_GOODS_PRICE"] <= 261900.0), 1, 0)
        X_new["AMT_GOODS_PRICE:261900.0–372600.0"] = np.where((X["AMT_GOODS_PRICE"] > 261900.0) & (X["AMT_GOODS_PRICE"] <= 372600.0), 1, 0)
        X_new["AMT_GOODS_PRICE:372600.0–483300.0"] = np.where((X["AMT_GOODS_PRICE"] > 372600.0) & (X["AMT_GOODS_PRICE"] <= 483300.0), 1, 0)
        X_new["AMT_GOODS_PRICE:483300.0–704700.0"] = np.where((X["AMT_GOODS_PRICE"] > 483300.0) & (X["AMT_GOODS_PRICE"] <= 704700.0), 1, 0)
        X_new["AMT_GOODS_PRICE:704700.0–1036800.0"] = np.where((X["AMT_GOODS_PRICE"] > 704700.0) & (X["AMT_GOODS_PRICE"] <= 1036800.0), 1, 0)
        X_new["AMT_GOODS_PRICE:1036800.0–2254500.0"] = np.where((X["AMT_GOODS_PRICE"] > 1036800.0) & (X["AMT_GOODS_PRICE"] <= 2254500.0), 1, 0)
        X_new["AMT_GOODS_PRICE:>2254500.0"] = np.where((X["AMT_GOODS_PRICE"] > 2254500.0), 1, 0)

        X_new["REGION_POPULATION_RELATIVE:<=0.00751"] = np.where((X["REGION_POPULATION_RELATIVE"] <= 0.00751), 1, 0)
        X_new["REGION_POPULATION_RELATIVE:0.00751–0.022"] = np.where((X["REGION_POPULATION_RELATIVE"] > 0.00751) & (X["REGION_POPULATION_RELATIVE"] <= 0.022), 1, 0)
        X_new["REGION_POPULATION_RELATIVE:0.022–0.0364"] = np.where((X["REGION_POPULATION_RELATIVE"] > 0.022) & (X["REGION_POPULATION_RELATIVE"] <= 0.0364), 1, 0)
        X_new["REGION_POPULATION_RELATIVE:0.0364–0.0725"] = np.where((X["REGION_POPULATION_RELATIVE"] > 0.0364) & (X["REGION_POPULATION_RELATIVE"] <= 0.0725), 1, 0)
        X_new["REGION_POPULATION_RELATIVE:>0.0725"] = np.where((X["REGION_POPULATION_RELATIVE"] > 0.0725), 1, 0)

        X_new["LOAN_DURATION:missing"] = np.where(X["LOAN_DURATION"].isnull(), 1, 0)
        X_new["LOAN_DURATION:<=10.148"] = np.where((X["LOAN_DURATION"] <= 10.148), 1, 0)
        X_new["LOAN_DURATION:10.148–14.37"] = np.where((X["LOAN_DURATION"] > 10.148) & (X["LOAN_DURATION"] <= 14.37), 1, 0)
        X_new["LOAN_DURATION:14.37–21.758"] = np.where((X["LOAN_DURATION"] > 14.37) & (X["LOAN_DURATION"] <= 21.758), 1, 0)
        X_new["LOAN_DURATION:21.758–28.092"] = np.where((X["LOAN_DURATION"] > 21.758) & (X["LOAN_DURATION"] <= 28.092), 1, 0)
        X_new["LOAN_DURATION:28.092–31.258"] = np.where((X["LOAN_DURATION"] > 28.092) & (X["LOAN_DURATION"] <= 31.258), 1, 0)
        X_new["LOAN_DURATION:31.258–39.702"] = np.where((X["LOAN_DURATION"] > 31.258) & (X["LOAN_DURATION"] <= 39.702), 1, 0)
        X_new["LOAN_DURATION:>39.702"] = np.where((X["LOAN_DURATION"] > 39.702), 1, 0)

        X_new.drop(columns = ref_categories, inplace = True)
        return X_new
    
def evaluation(df, threshold, y_actual, y_predicted, y_proba, recall, precision):
    
    results = {
        "Threshold": threshold,
        "Accuracy": metrics.accuracy_score(df[y_actual], df[y_predicted]),
        "Precision": metrics.precision_score(df[y_actual], df[y_predicted]),
        "Recall": metrics.recall_score(df[y_actual], df[y_predicted]),
        "F1": metrics.f1_score(df[y_actual], df[y_predicted]),
        "AUROC": metrics.roc_auc_score(df[y_actual], df[y_proba]),
        "Gini": metrics.roc_auc_score(df[y_actual], df[y_proba]) * 2 - 1,
        "AUCPR": metrics.auc(recall, precision),
    }
    return results

def confusion_matrix_mod(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2%", ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')

    True_Pos = cm[1, 1]
    True_Neg = cm[0, 0]
    False_Pos = cm[0, 1]
    False_Neg = cm[1, 0]

    plt.show()

    return True_Pos, True_Neg, False_Pos, False_Neg
    