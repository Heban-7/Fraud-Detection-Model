# import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# load data
def load_data(filepath1, filepath2, filepath3):
    fraud_data = pd.read_csv(filepath1)
    ip_address = pd.read_csv(filepath2)
    credit_card = pd.read_csv(filepath3)

    return fraud_data, ip_address, credit_card

def data_summary(df):
    """
    Provides a detailed summary of the dataset including shape, 
    statistical description for numerical and categorical columns, 
    and column information.
    """
    print(f"Shape of the dataset: {df.shape}\n")

    print("Numerical Column Description:\n")
    print(df.describe())

    # Unique values per column
    unique_counts = df.nunique()
    print("\n\nUnique Values per Column:\n")
    print(unique_counts)

    print("\n\nDataset Information:\n")
    print(df.info())


def get_fraud_data(fraud_data_df):
    fraud_df = fraud_data_df.copy()

    # Convert to datetime
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

    # Time difference features
    time_diff = fraud_df['purchase_time'] - fraud_df['signup_time']
    fraud_df['time_to_first_purchase_hours'] = time_diff.dt.total_seconds() / 3600

    return fraud_df



def purchase_value_visualization(fraud_df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2,2,1)
    sns.histplot(fraud_df['purchase_value'], kde=True, color='skyblue', edgecolor='black')
    plt.title(" Distrubution of Purchase Value")

    plt.subplot(2,2,2)
    plt.boxplot(fraud_df['purchase_value'])
    plt.title('Box Plot of Purchase Value')

    plt.subplot(2,2,3)
    sns.boxplot(x=fraud_df['class'], y=fraud_df['purchase_value'])
    plt.xlabel("Fraud Class")
    plt.ylabel("Purchase Value")
    plt.title("Purchase Value vs. Fraud Class")
    plt.xticks([0, 1], ["Not Fraud", "Fraud"])


    plt.subplot(2,2,4)
    sns.histplot(data=fraud_df, x='purchase_value', hue='class', multiple='stack', palette='Set1')
    plt.title('Purchase Value Distribution By Fraud Class')
    plt.show()

def age_visualization(fraud_df):
    plt.figure(figsize=(15,10))

    plt.subplot(2,2,1)
    sns.histplot(fraud_df['age'], kde=True, color='purple')
    plt.xlabel("Age")
    plt.title('Age Distribution')

    plt.subplot(2,2,2)
    sns.boxplot(fraud_df['age'])
    plt.ylabel("Age")
    plt.title('Age Box Plot')

    plt.subplot(2,2,3)
    sns.boxplot(x=fraud_df['class'], y=fraud_df['age'])
    plt.xlabel("Fraud Class")
    plt.ylabel("Age")
    plt.title("Age vs. Fraud Class")
    plt.xticks([0, 1], ["Not Fraud", "Fraud"])

    plt.subplot(2,2,4)
    sns.histplot(data=fraud_df, x='age', hue='class', multiple='stack', palette='Set2')
    plt.title('Age Distribution by Fraud Class')
    plt.show()

def categorical_value_visualization(fraud_df):
    plt.figure(figsize=(12,15))
    
    plt.subplot(3, 2, 1)
    sns.countplot(data=fraud_df, x = 'browser', palette='Set1')
    plt.title('Browser Count Distribution')

    plt.subplot(3,2,2)
    sns.countplot(data=fraud_df, x='browser', hue='class')
    plt.title('Browser vs. Fraud Class')
    plt.legend(title ="Fraud Class")

    plt.subplot(3,2,3)
    sns.countplot(data=fraud_df, x='source', palette='Set2')
    plt.title("Source Count Distribution")

    plt.subplot(3,2,4)
    sns.countplot(data=fraud_df, x='source', hue='class')
    plt.title('Source vs. Fraud Class')
    plt.legend(title='Fraud Class')

    plt.subplot(3,2,5)
    sns.countplot(data=fraud_df, x='sex', palette='Set3')
    plt.title('Sex Count Distribution')
    plt.xticks(['M', 'F'], ['Male', "Female"])

    plt.subplot(3,2,6)
    sns.countplot(data=fraud_df, x='sex', hue='class')
    plt.title('Sex vs. Fraud Class')
    plt.legend(title='Fraud Class')
    plt.xticks(['M', 'F'], ['Male', "Female"])

    plt.show()

# Histogram Distribution of time to first purchase
def time_to_first_purchase_visualization(fraud_df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    sns.histplot(data=fraud_df, x='time_to_first_purchase_hours', hue='class', multiple='stack', kde=True)
    plt.xlabel("Time to First Purchase (hours)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Time to First Purchase")

    plt.subplot(1,2,2)
    sns.kdeplot(fraud_df[fraud_df['class']== 0]['time_to_first_purchase_hours'], label='Not Fraud', shade=True, color='blue')
    sns.kdeplot(fraud_df[fraud_df['class']== 1]['time_to_first_purchase_hours'], label='Fraud', shade=True, color='red')
    plt.xlabel('Time To First Purchase (Hours)')
    plt.ylabel('Density')
    plt.title("Distribution of Time to First Purchase by Fraud Class")
    plt.legend()
    plt.show()


def signup_purchase_time_trend(fraud_df):
    df =fraud_df.copy()
    df['signup_date'] = df['signup_time'].dt.date
    signup_counts = df.groupby('signup_date').size()


    df['purchase_date'] = df['purchase_time'].dt.date
    purchase_counts = df.groupby('purchase_date').size()

    plt.figure(figsize=(12, 6))
    signup_counts.plot(label="Signups", linestyle='--', marker='o')
    purchase_counts.plot(label="Purchases", linestyle='-', marker='o')
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title("Signups & Purchases Over Time")
    plt.legend()
    plt.show()

# relationship between signup_time and purchase_time
def relation_signup_purchase_time(fraud_df):
    plt.figure(figsize=(10, 5))
    plt.scatter(fraud_df['signup_time'], fraud_df['purchase_time'], alpha=0.5, c=fraud_df['class'], cmap='coolwarm')
    plt.xlabel("Signup Time")
    plt.ylabel("Purchase Time")
    plt.title("Signup vs Purchase Time (Colored by Fraud Class)")
    plt.colorbar(label="Fraud Class")
    plt.show()

def merge_data(fraud_data_df, ip_address_df):
    # Convert IP addresses to numeric
    fraud = fraud_data_df.copy()
    fraud['ip_address'] = pd.to_numeric(fraud['ip_address'], errors='coerce')
    ip_address_df = ip_address_df.copy()
    ip_address_df['lower_bound_ip_address'] = pd.to_numeric(ip_address_df['lower_bound_ip_address'], errors='coerce')
    ip_address_df['upper_bound_ip_address'] = pd.to_numeric(ip_address_df['upper_bound_ip_address'], errors='coerce')

    # Merge with IP-country mapping
    merged_df = pd.merge_asof(
        fraud.sort_values('ip_address'),
        ip_address_df.sort_values('lower_bound_ip_address'),
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Filter valid IP ranges and add country frequency
    merged_df = merged_df[
        (merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])
    ]
    merged_df.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    
    return merged_df


def feature_engineering(df):
    df = df.copy()
        # Convert to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Time difference features
    time_diff = df['purchase_time'] - df['signup_time']
    df['time_to_first_purchase_days'] = time_diff.dt.total_seconds() / (24*3600)
    df['time_to_first_purchase_hours'] = time_diff.dt.total_seconds() / 3600
    
    # Immediate purchase flag
    df['immediate_purchase'] = (df['time_to_first_purchase_hours'] < 1).astype(int)

    # Datetime feature extraction
    for prefix in ['signup', 'purchase']:
        dt_col = f'{prefix}_time'
        df[f'{prefix}_month'] = df[dt_col].dt.month
        df[f'{prefix}_day'] = df[dt_col].dt.day
        df[f'{prefix}_hour'] = df[dt_col].dt.hour
        df[f'{prefix}_dayofweek'] = df[dt_col].dt.dayofweek
        df[f'{prefix}_is_weekend'] = df[f'{prefix}_dayofweek'].isin([5,6]).astype(int)

    # Time of day categorization
    def time_of_day(hour):
        if 0 <= hour < 6: return 'night'
        elif hour < 12: return 'morning'
        elif hour < 18: return 'afternoon'
        else: return 'evening'
    
    for prefix in ['purchase']:
        df[f'{prefix}_timeofday'] = df[f'{prefix}_hour'].apply(time_of_day)

    # Country frequency 
    df['country_freq'] = df.groupby('country')['country'].transform('count')
    
    # Age binning
    df['age_group'] = pd.cut(df['age'],
                            bins=[0, 18, 25, 35, 50, 100],
                            labels=['_18', '18_25', '26_35', '36_50', '50_'])

    # Transaction value transformations
    df['log_purchase_value'] = np.log1p(df['purchase_value'])
    df['purchase_value_high'] = (df['purchase_value'] > df['purchase_value'].quantile(0.95)).astype(int)

    # Aggregated features
    for entity in ['device_id', 'ip_address', 'user_id']:
        # Transaction counts
        df[f'{entity}_tx_count'] = df.groupby(entity)[entity].transform('count')
        
        # Unique users per device/ip
        if entity != 'user_id':
            df[f'{entity}_unique_users'] = df.groupby(entity)['user_id'].transform('nunique')

    # Drop original columns
    cols_to_drop = ['signup_time', 'purchase_time', 'user_id', 
                   'device_id', 'ip_address', 'time_to_first_purchase_hours']
    df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

    return df


def numerical_scaler(df):
    numerical_cols = [
        'purchase_value', 'age', 'time_to_first_purchase_days',
        'log_purchase_value', 'country_freq', 'device_id_tx_count',
        'ip_address_tx_count', 'user_id_tx_count', 'device_id_unique_users',
        'ip_address_unique_users', 'signup_month', 'signup_day', 'signup_hour',
        'purchase_month', 'purchase_day', 'purchase_hour'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def categorical_encoder(df):
    # Convert remaining strings to categories
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Cinvert country into numerical using LabelEncoder
    df['country'] = label_encoder.fit_transform(df['country'])

    # Get categorical columns
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    
    # One-hot encode all categorical features
    df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_')
    
    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df