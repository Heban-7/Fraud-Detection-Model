# import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
encoder = LabelEncoder()
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

    # Convert signup time and purchase time to pandas datetime format
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

    # Create time difference feature between signup and first purchase time
    fraud_df['time_to_first_purchase'] = ((fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()) / (24 * 3600)  # Convert seconds to days

    # Move column next to purchase time
    fraud_df.insert(
        fraud_df.columns.get_loc('purchase_time') + 1, 
        'time_to_first_purchase', 
        fraud_df.pop('time_to_first_purchase')
    )

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
    sns.histplot(data=fraud_df, x='time_to_first_purchase', hue='class', multiple='stack', kde=True)
    plt.xlabel("Time to First Purchase (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Time to First Purchase")

    plt.subplot(1,2,2)
    sns.kdeplot(fraud_df[fraud_df['class']== 0]['time_to_first_purchase'] /3600, label='Not Fraud', shade=True, color='blue')
    sns.kdeplot(fraud_df[fraud_df['class']== 1]['time_to_first_purchase'] /3600, label='Fraud', shade=True, color='red')
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

def merge_data(fraud_df, ip_address_df):
    fraud = fraud_df.copy()
    ip = ip_address_df.copy()
    # Ensure data types are numeric for range comparison
    fraud['ip_address'] = pd.to_numeric(fraud['ip_address'], errors='coerce')
    ip['lower_bound_ip_address'] = pd.to_numeric(ip['lower_bound_ip_address'], errors='coerce')
    ip['upper_bound_ip_address'] = pd.to_numeric(ip['upper_bound_ip_address'], errors='coerce')

    # Sort both datasets for asof merge
    fraud = fraud.sort_values(by='ip_address')
    ip = ip.sort_values(by='lower_bound_ip_address')

    # Merge datasets by range
    merged_df = pd.merge_asof(
        fraud, 
        ip, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address', 
        direction='backward'
    )

    # Filter rows where ip_address falls within valid ranges
    merged_df = merged_df[
        (merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])
    ]

    # Select relevant columns
    merged_df = merged_df.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)

    # Display the merged result
    return merged_df


def feature_engineering(fraud_data_df, ip_address_df):

    fraud_df = get_fraud_data(fraud_data_df)

    df = merge_data(fraud_df, ip_address_df)

    # Convert columns to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Extract year, month, day, and hour from time columns
    df['signup_year'] = df['signup_time'].dt.year
    df['signup_month'] = df['signup_time'].dt.month
    df['signup_day'] = df['signup_time'].dt.day
    df['signup_hour'] = df['signup_time'].dt.hour

    df['purchase_year'] = df['purchase_time'].dt.year
    df['purchase_month'] = df['purchase_time'].dt.month
    df['purchase_day'] = df['purchase_time'].dt.day
    df['purchase_hour'] = df['purchase_time'].dt.hour

    # drop uncessarly columns for model training
    df.drop(['signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address', 'signup_year', 'purchase_year'], axis =1, inplace = True)

    return df

def numerical_scaler(df):
    numerical_cols = [
    "purchase_value", "age", "time_to_first_purchase",
    "signup_month", "signup_day", "signup_hour",
    "purchase_month", "purchase_day", "purchase_hour"
    ]
    # Apply StandardScaler
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def categorical_encoder(df):
    # Convert all object columns to categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Convert categorical columns into numerical using one-hot-encoding
    df = pd.get_dummies(df, columns=['source', 'browser', 'sex'], prefix= ['source', 'browser', 'sex'])

    # Convert all dummy columns (which are uint8) into int
    df = df.astype({col: int for col in df.select_dtypes(include=['bool']).columns})
    
    df['country'] = encoder.fit_transform(df['country'])

    return df
