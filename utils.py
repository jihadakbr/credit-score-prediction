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
from sklearn import model_selection, pipeline, feature_selection, metrics, set_config, linear_model, base

#########################################################################################################################
#########################################################################################################################

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

def outliers_graph(data, cols):
    num_cols = len(cols)
    num_rows = math.ceil(num_cols / 8)
    
    fig, axs = plt.subplots(num_rows, 8, figsize=(24, 6*num_rows))
    fig.tight_layout(w_pad=5.0, h_pad=3.0)
    axs = axs.flatten()
    
    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted")
    
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col], ax=axs[i], color=custom_palette[i % len(custom_palette)])
        axs[i].set_ylabel(col, fontsize=14)
        axs[i].grid(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
    
    for j in range(num_cols, num_rows*8):
        fig.delaxes(axs[j])
    
    return fig, axs

def outliers_graph_mod(data, cols):
    num_cols = len(cols)
    num_rows = math.ceil(num_cols / 4)
    
    fig, axs = plt.subplots(num_rows, 4, figsize=(24, 6*num_rows))
    fig.tight_layout(w_pad=5.0, h_pad=3.0)
    axs = axs.flatten()
    
    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted")
    
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col], ax=axs[i], color=custom_palette[i % len(custom_palette)])
        axs[i].set_ylabel(col, fontsize=14)
        axs[i].grid(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
    
    for j in range(num_cols, num_rows*4):
        fig.delaxes(axs[j])
    
    return fig, axs

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
    
def cat_dist(data, col_x, col_target, target_1_label, target_0_label, colour_number):
    palette = ["husl","hls","Spectral","coolwarm"]
    sns.set_style("whitegrid")
    ax = sns.countplot(data=data, x=col_x, hue=col_target, palette=palette[colour_number])
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_0_label, target_1_label])
    plt.grid(False)

    for i in ax.patches:
        ax.text(i.get_x()+0.05, i.get_height()+1000, int(i.get_height()), fontsize=11)

def cat_dist_top5(data, col_x, col_target, target_1_label, target_0_label, colour_number):
    palette = ["husl","hls","Spectral","coolwarm"]
    sns.set_style("whitegrid")
    top5 = data[col_x].value_counts().index[:5]
    ax = sns.countplot(data=data[data[col_x].isin(top5)], x=col_x, hue=col_target, 
                       palette=palette[colour_number], order=top5)
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_0_label, target_1_label])
    plt.grid(False)
    plt.title(f"Top 5 {col_x} categories", fontsize=18, pad=20)

    for i in ax.patches:
        ax.text(i.get_x(), i.get_height()+1000, int(i.get_height()), fontsize=11)
    
def target_dist(data, col_target, target_1_label, target_0_label):
    mpl.rcParams['font.size'] = 11
    r = data.groupby(col_target)[col_target].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[target_0_label, target_1_label], radius=1.5, autopct='%1.1f%%', shadow=True, startangle=45,
           colors=['#66b3ff', '#ff9999'])
    ax.set_aspect('equal')
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

ref_categories = ["NAME_CONTRACT_TYPE:Cash loans","FLAG_OWN_CAR:N",
                  "EXT_SOURCE_2:>0.855","EXT_SOURCE_3:>0.894","EXT_SOURCE_1:>0.943","DAYS_EMPLOYED:>0.0","REGION_RATING_CLIENT:1",
                  "AGE_IN_YEARS:>62","DAYS_LAST_PHONE_CHANGE:>0.0","REG_CITY_NOT_WORK_CITY:1","AMT_CREDIT:>4050000.0","FLAG_DOCUMENT_3:1",
                  "DAYS_ID_PUBLISH:>0.0","DAYS_REGISTRATION:>0.0"]
                  
class WoE_Binning(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, X):
        self.X = X
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_new = X.loc[:, "NAME_CONTRACT_TYPE:Cash loans":"NAME_CONTRACT_TYPE:Revolving loans"]
        X_new = pd.concat([X_new, X.loc[:, "FLAG_OWN_CAR:N":"FLAG_OWN_CAR:Y"]], axis = 1)
        
        X_new["EXT_SOURCE_2:<=0.171"] = np.where((X["EXT_SOURCE_2"] <= 0.171), 1, 0)
        X_new["EXT_SOURCE_2:0.171–0.342"] = np.where((X["EXT_SOURCE_2"] > 0.171) & (X["EXT_SOURCE_2"] <= 0.342), 1, 0)
        X_new["EXT_SOURCE_2:0.342–0.456"] = np.where((X["EXT_SOURCE_2"] > 0.342) & (X["EXT_SOURCE_2"] <= 0.456), 1, 0)
        X_new["EXT_SOURCE_2:0.456–0.627"] = np.where((X["EXT_SOURCE_2"] > 0.456) & (X["EXT_SOURCE_2"] <= 0.627), 1, 0)
        X_new["EXT_SOURCE_2:0.627–0.855"] = np.where((X["EXT_SOURCE_2"] > 0.627) & (X["EXT_SOURCE_2"] <= 0.855), 1, 0)
        X_new["EXT_SOURCE_2:>0.855"] = np.where((X["EXT_SOURCE_2"] > 0.855), 1, 0)

        X_new["EXT_SOURCE_3:<=0.179"] = np.where((X["EXT_SOURCE_3"] <= 0.179), 1, 0)
        X_new["EXT_SOURCE_3:0.179–0.417"] = np.where((X["EXT_SOURCE_3"] > 0.179) & (X["EXT_SOURCE_3"] <= 0.417), 1, 0)
        X_new["EXT_SOURCE_3:0.417–0.596"] = np.where((X["EXT_SOURCE_3"] > 0.417) & (X["EXT_SOURCE_3"] <= 0.596), 1, 0)
        X_new["EXT_SOURCE_3:0.596–0.715"] = np.where((X["EXT_SOURCE_3"] > 0.596) & (X["EXT_SOURCE_3"] <= 0.715), 1, 0)
        X_new["EXT_SOURCE_3:0.715–0.894"] = np.where((X["EXT_SOURCE_3"] > 0.715) & (X["EXT_SOURCE_3"] <= 0.894), 1, 0)
        X_new["EXT_SOURCE_3:>0.894"] = np.where((X["EXT_SOURCE_3"] > 0.894), 1, 0)
        
        X_new["EXT_SOURCE_1:<=0.262"] = np.where((X["EXT_SOURCE_1"] <= 0.262), 1, 0)
        X_new["EXT_SOURCE_1:0.262–0.386"] = np.where((X["EXT_SOURCE_1"] > 0.262) & (X["EXT_SOURCE_1"] <= 0.386), 1, 0)
        X_new["EXT_SOURCE_1:0.386–0.571"] = np.where((X["EXT_SOURCE_1"] > 0.386) & (X["EXT_SOURCE_1"] <= 0.571), 1, 0)
        X_new["EXT_SOURCE_1:0.571–0.695"] = np.where((X["EXT_SOURCE_1"] > 0.571) & (X["EXT_SOURCE_1"] <= 0.695), 1, 0)
        X_new["EXT_SOURCE_1:0.695–0.943"] = np.where((X["EXT_SOURCE_1"] > 0.695) & (X["EXT_SOURCE_1"] <= 0.943), 1, 0)
        X_new["EXT_SOURCE_1:>0.943"] = np.where((X["EXT_SOURCE_1"] > 0.943), 1, 0)
        
        X_new["DAYS_EMPLOYED:<=-4478.0"] = np.where((X["DAYS_EMPLOYED"] <= -4478.0), 1, 0)
        X_new["DAYS_EMPLOYED:-4478.0–0.0"] = np.where((X["DAYS_EMPLOYED"] > -4478.0) & (X["DAYS_EMPLOYED"] <= 0.0), 1, 0)
        X_new["DAYS_EMPLOYED:>0.0"] = np.where((X["DAYS_EMPLOYED"] > 0.0), 1, 0)
        
        X_new["REGION_RATING_CLIENT:1"] = np.where((X["REGION_RATING_CLIENT"] == 1), 1, 0)
        X_new["REGION_RATING_CLIENT:2"] = np.where((X["REGION_RATING_CLIENT"] == 2), 1, 0)
        X_new["REGION_RATING_CLIENT:3"] = np.where((X["REGION_RATING_CLIENT"] == 3), 1, 0)
        
        X_new["AGE_IN_YEARS:<=25"] = np.where((X["AGE_IN_YEARS"] <= 25), 1, 0)
        X_new["AGE_IN_YEARS:25–30"] = np.where((X["AGE_IN_YEARS"] > 25) & (X["AGE_IN_YEARS"] <= 30), 1, 0)
        X_new["AGE_IN_YEARS:30–34"] = np.where((X["AGE_IN_YEARS"] > 30) & (X["AGE_IN_YEARS"] <= 34), 1, 0)
        X_new["AGE_IN_YEARS:34–38"] = np.where((X["AGE_IN_YEARS"] > 34) & (X["AGE_IN_YEARS"] <= 38), 1, 0)
        X_new["AGE_IN_YEARS:38–45"] = np.where((X["AGE_IN_YEARS"] > 38) & (X["AGE_IN_YEARS"] <= 45), 1, 0)
        X_new["AGE_IN_YEARS:45–52"] = np.where((X["AGE_IN_YEARS"] > 45) & (X["AGE_IN_YEARS"] <= 52), 1, 0)
        X_new["AGE_IN_YEARS:52–62"] = np.where((X["AGE_IN_YEARS"] > 52) & (X["AGE_IN_YEARS"] <= 62), 1, 0)
        X_new["AGE_IN_YEARS:>62"] = np.where((X["AGE_IN_YEARS"] > 62), 1, 0)
        
        X_new["DAYS_LAST_PHONE_CHANGE:<=-2289.067"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] <= -2289.067), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-2289.067–(-1144.533)"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -2289.067) & (X["DAYS_LAST_PHONE_CHANGE"] <= -1144.533), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:-1144.533–0.0"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > -1144.533) & (X["DAYS_LAST_PHONE_CHANGE"] <= 0.0), 1, 0)
        X_new["DAYS_LAST_PHONE_CHANGE:>0.0"] = np.where((X["DAYS_LAST_PHONE_CHANGE"] > 0.0), 1, 0)

        X_new["REG_CITY_NOT_WORK_CITY:0"] = np.where((X["REG_CITY_NOT_WORK_CITY"] == 0), 1, 0)
        X_new["REG_CITY_NOT_WORK_CITY:1"] = np.where((X["REG_CITY_NOT_WORK_CITY"] == 1), 1, 0)
        
        X_new["AMT_CREDIT:<=40995.0"] = np.where((X["AMT_CREDIT"] <= 40995.0), 1, 0)
        X_new["AMT_CREDIT:40995.0–1189285.714"] = np.where((X["AMT_CREDIT"] > 40995.0) & (X["AMT_CREDIT"] <= 1189285.714), 1, 0)
        X_new["AMT_CREDIT:1189285.714–4050000.0"] = np.where((X["AMT_CREDIT"] > 1189285.714) & (X["AMT_CREDIT"] <= 4050000.0), 1, 0)
        X_new["AMT_CREDIT:>4050000.0"] = np.where((X["AMT_CREDIT"] > 4050000.0), 1, 0)
        
        X_new["FLAG_DOCUMENT_3:0"] = np.where((X["FLAG_DOCUMENT_3"] == 0), 1, 0)
        X_new["FLAG_DOCUMENT_3:1"] = np.where((X["FLAG_DOCUMENT_3"] == 1), 1, 0)
        
        X_new["DAYS_ID_PUBLISH:<=-4827.053"] = np.where((X["DAYS_ID_PUBLISH"] <= -4827.053), 1, 0)
        X_new["DAYS_ID_PUBLISH:-4827.053–(-4137.474)"] = np.where((X["DAYS_ID_PUBLISH"] > -4827.053) & (X["DAYS_ID_PUBLISH"] <= -4137.474), 1, 0)
        X_new["DAYS_ID_PUBLISH:-4137.474–(-3103.105)"] = np.where((X["DAYS_ID_PUBLISH"] > -4137.474) & (X["DAYS_ID_PUBLISH"] <= -3103.105), 1, 0)
        X_new["DAYS_ID_PUBLISH:-3103.105–(-1723.947)"] = np.where((X["DAYS_ID_PUBLISH"] > -3103.105) & (X["DAYS_ID_PUBLISH"] <= -1723.947), 1, 0)
        X_new["DAYS_ID_PUBLISH:-1723.947–0.0"] = np.where((X["DAYS_ID_PUBLISH"] > -1723.947) & (X["DAYS_ID_PUBLISH"] <= 0.0), 1, 0)
        X_new["DAYS_ID_PUBLISH:>0.0"] = np.where((X["DAYS_ID_PUBLISH"] > 0.0), 1, 0)
        
        X_new["DAYS_REGISTRATION:<=-9080.4"] = np.where((X["DAYS_REGISTRATION"] <= -9080.4), 1, 0)
        X_new["DAYS_REGISTRATION:-9080.4–(-4540.2)"] = np.where((X["DAYS_REGISTRATION"] > -9080.4) & (X["DAYS_REGISTRATION"] <= -4540.2), 1, 0)
        X_new["DAYS_REGISTRATION:-4540.2–0.0"] = np.where((X["DAYS_REGISTRATION"] > -4540.2) & (X["DAYS_REGISTRATION"] <= 0.0), 1, 0)      
        X_new["DAYS_REGISTRATION:>0.0"] = np.where((X["DAYS_REGISTRATION"] > 0.0), 1, 0)
        
        X_new.drop(columns = ref_categories, inplace = True)
        return X_new
    
def evaluation(df,threshold,y_actual,y_predicted,y_proba,recall,precision):
    results = {
        "Threshold": threshold,
        "Accuracy":  metrics.accuracy_score(df[y_actual], df[y_predicted]),
        "Precision": metrics.precision_score(df[y_actual], df[y_predicted]),
        "Recall":    metrics.recall_score(df[y_actual], df[y_predicted]),
        "F1":        metrics.f1_score(df[y_actual], df[y_predicted]),
        "AUROC":     metrics.roc_auc_score(df[y_actual], df[y_proba]),
        "Gini":      metrics.roc_auc_score(df[y_actual], df[y_proba]) * 2 - 1,
        "AUCPR":     metrics.auc(recall, precision)
    }
    return results

def confusion_matrix_mod(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2%", ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')

    True_Pos = cm[1, 1]
    False_Neg = cm[1, 0]

    plt.show()

    return True_Pos, False_Neg
    