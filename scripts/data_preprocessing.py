# import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

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
    fraud_df['time_to_first_purchase'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()

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
    plt.title('Source Count Distribution')

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


def feature_engineering(df):
    merged_df =df.copy()
    # Convert columns to datetime
    merged_df['signup_time'] = pd.to_datetime(merged_df['signup_time'], errors='coerce')
    merged_df['purchase_time'] = pd.to_datetime(merged_df['purchase_time'], errors='coerce')

    # Extract year, month, day, and hour from time columns
    merged_df['signup_year'] = merged_df['signup_time'].dt.year
    merged_df['signup_month'] = merged_df['signup_time'].dt.month
    merged_df['signup_day'] = merged_df['signup_time'].dt.day
    merged_df['signup_hour'] = merged_df['signup_time'].dt.hour

    merged_df['purchase_year'] = merged_df['purchase_time'].dt.year
    merged_df['purchase_month'] = merged_df['purchase_time'].dt.month
    merged_df['purchase_day'] = merged_df['purchase_time'].dt.day
    merged_df['purchase_hour'] = merged_df['purchase_time'].dt.hour

    merged_df['days_between_signup_purchase'] = (merged_df['purchase_time'] - merged_df['signup_time']).dt.days
    merged_df['transaction_velocity'] = 1 / (merged_df['days_between_signup_purchase'].replace(0, 1))

    # age bining 
    age_bins = [0, 18, 35, 55, 100]
    age_labels = ['Youth', 'Adult', 'Middle_Aged', 'Senior']
    merged_df['age_group'] = pd.cut(merged_df['age'], bins=age_bins, labels=age_labels)

    # drop uncessarly columns for model training
    merged_df.drop(['signup_time', 'purchase_time', 'user_id', 'device_id'], axis =1, inplace = True)

    return merged_df


def categorical_encoder(df):
    # Convert all object columns to categorical
    categorical_col = []
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        categorical_col.append(col)

    df = pd.get_dummies(df, columns=['source', 'browser', 'sex', 'age_group'], prefix= ['source', 'browser', 'sex', 'age_group'])
    
    df['country'] = encoder.fit_transform(df['country'])

    return df
