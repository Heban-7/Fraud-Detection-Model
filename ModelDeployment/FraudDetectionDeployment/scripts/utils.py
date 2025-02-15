import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize scaler and label encoder
scaler = StandardScaler()
label_encoder = LabelEncoder()

def feature_engineering(df):
    """
    Performs feature engineering on the input DataFrame.
    """
    df = df.copy()

    # Convert to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Time difference features
    time_diff = df['purchase_time'] - df['signup_time']
    df['time_to_first_purchase_days'] = time_diff.dt.total_seconds() / (24 * 3600)
    df['time_to_first_purchase_hours'] = time_diff.dt.total_seconds() / 3600

    # Immediate purchase flag
    df['immediate_purchase'] = (df['time_to_first_purchase_hours'] < 1).astype(int)

    # Datetime feature extraction for signup and purchase
    for prefix in ['signup', 'purchase']:
        dt_col = f'{prefix}_time'
        df[f'{prefix}_month'] = df[dt_col].dt.month
        df[f'{prefix}_day'] = df[dt_col].dt.day
        df[f'{prefix}_hour'] = df[dt_col].dt.hour
        df[f'{prefix}_dayofweek'] = df[dt_col].dt.dayofweek
        df[f'{prefix}_is_weekend'] = df[f'{prefix}_dayofweek'].isin([5, 6]).astype(int)

    # Time of day categorization for purchase hour
    def time_of_day(hour):
        if 0 <= hour < 6:
            return 'night'
        elif hour < 12:
            return 'morning'
        elif hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    df['purchase_timeofday'] = df['purchase_hour'].apply(time_of_day)

    # Country frequency
    df['country_freq'] = df.groupby('country')['country'].transform('count')

    # Age binning
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 18, 25, 35, 50, 100],
        labels=['_18', '18_25', '26_35', '36_50', '50_']
    )

    # Transaction value transformations
    df['log_purchase_value'] = np.log1p(df['purchase_value'])
    df['purchase_value_high'] = (df['purchase_value'] > df['purchase_value'].quantile(0.95)).astype(int)

    # Aggregated features per entity
    for entity in ['device_id', 'ip_address', 'user_id']:
        # Transaction counts
        df[f'{entity}_tx_count'] = df.groupby(entity)[entity].transform('count')
        # Unique users per device/ip (skip for user_id)
        if entity != 'user_id':
            df[f'{entity}_unique_users'] = df.groupby(entity)['user_id'].transform('nunique')

    # Drop columns not used in prediction
    cols_to_drop = [
        'signup_time', 'purchase_time', 'user_id',
        'device_id', 'ip_address', 'time_to_first_purchase_hours'
    ]
    df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

    return df

def numerical_scaler(df):
    """
    Scales numerical features.
    """
    numerical_cols = [
        'purchase_value', 'age', 'time_to_first_purchase_days',
        'log_purchase_value', 'country_freq', 'device_id_tx_count',
        'ip_address_tx_count', 'user_id_tx_count', 'device_id_unique_users',
        'ip_address_unique_users', 'signup_month', 'signup_day', 'signup_hour',
        'purchase_month', 'purchase_day', 'purchase_hour'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    # In production, use scaler.transform() with a pre-fitted scaler.
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def categorical_encoder(df):
    """
    Encodes categorical features.
    """
    # Convert object types to category
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    # Convert 'country' to numerical using LabelEncoder
    df['country'] = label_encoder.fit_transform(df['country'])

    # Get list of categorical columns
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_')

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def preprocess_input(df, expected_columns):
    """
    Applies feature engineering, scaling, and encoding to the input DataFrame
    """
    df = feature_engineering(df)
    df = numerical_scaler(df)
    df = categorical_encoder(df)
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

