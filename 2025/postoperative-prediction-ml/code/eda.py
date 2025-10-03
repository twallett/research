#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

df = pd.read_csv("data/LapGenSurgOnly_2022.csv")

INPUT_FEATURES = [
    "Age", "SEX", "RACE_NEW", "BMI", "INOUT", "ASACLAS", "CPT", 
    "DIABETES", "SMOKE", "FNSTATUS2", "HXCOPD", "ASCITES", "HXCHF", "HYPERMED",
    "DIALYSIS", "DISCANCR", "STEROID", "TRANSFUS"
]

LABEL_MAPPING = {
    "Age": "Age",
    "SEX": "Sex",
    "RACE_NEW": "Race",
    "BMI": "BMI",
    "INOUT": "Hospital Status",
    "ASACLAS": "ASA Classification",
    "CPT": "CPT",
    "DIABETES": "Diabetes",
    "SMOKE": "Smoke",
    "FNSTATUS2": "Functional Health Status",
    "HXCOPD": "History Pulmonary Disease",
    "ASCITES": "Ascites",
    "HXCHF": "History Congestive Heart Failure",
    "HYPERMED": "Hypertension",
    "DIALYSIS": "Dialysis",
    "DISCANCR": "Disseminated Cancer",
    "STEROID": "Steroid",
    "TRANSFUS": "Transfusion"
}

df_clean = df[INPUT_FEATURES].dropna().copy()

df_clean["Age"] = df_clean["Age"].round().astype("object")
df_clean["BMI"] = df_clean["BMI"].round().astype("object")

df_encoded = df_clean.copy()
label_encoders = {}

for col in INPUT_FEATURES:
    if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

def cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

def cramers_v_matrix(df, cols):
    """Generate a Cramer's V matrix for the specified categorical columns."""
    matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(confusion_matrix)
    
    return matrix

categorical_cols = [col for col in INPUT_FEATURES if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category']

cramers_matrix = cramers_v_matrix(df_clean, categorical_cols)

cramers_matrix.rename(index=LABEL_MAPPING, columns=LABEL_MAPPING, inplace=True)

plt.figure(figsize=(12, 10))
sns.heatmap(cramers_matrix.round(2), annot=True, vmin=0, vmax=1,
            xticklabels=True, yticklabels=True)
plt.title("Cramér's V Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation-matrix.pdf")
plt.show()
# %%