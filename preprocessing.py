import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Data cleaning
    cols_to_drop = [
        "patientID", "hospAdatetime", "hospDdatetime", "icuAdatetime", "icuDdatetime",
        "apache4", "DISCHDISPOSITIONTXT", "teamtransfer", "BinDaysfromHostoICUAdmit"
    ]
    df = df.drop(columns=cols_to_drop)

    df = df[df['icudeath'] == 0]
    df = df[df['hospdeath'] == 0]
    # df = df[df['INITIALADMIT'] == 1]
    df = df.drop(columns=['icudeath'])
    df = df.drop(columns=['hospdeath'])
    # df = df.drop(columns=['INITIALADMIT'])

    # Apply KNN imputation
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imputer = KNNImputer(n_neighbors=3)
    imputed_values = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = imputed_values

    # Apply ordinal encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    # Impute missing values in categorical columns
    imputer = KNNImputer(n_neighbors=3)
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    # Combine 'readmit72' and 'readmit72_next' into a single target column
    df['readmit72'] = df[['readmit72', 'readmit72_next']].any(axis=1)
    df = df.drop(columns=['readmit72_next'])

    # Identify binary columns (assuming binary columns contain only 0 and 1)
    binary_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

    # Identify non-binary columns
    non_binary_cols = [col for col in df.columns if col not in binary_cols]

    # Apply MinMaxScaler to non-binary columns
    scaler = MinMaxScaler()
    df[non_binary_cols] = scaler.fit_transform(df[non_binary_cols])

    return df

# Example usage
# df = pd.read_csv("patient_data.csv")
# imputed_df = preprocess_data(df)
# print("Number of missing values after imputation:", imputed_df.isna().sum().sum())
# print(imputed_df.columns)
