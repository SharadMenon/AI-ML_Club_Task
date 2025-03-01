import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Reading the dataset
df = pd.read_csv('air_quality_index.csv') 
# Replace -200 with NaN for easier handling
df.replace(-200, np.nan, inplace=True)

# Performing linear interpolation for missing values
df.interpolate(method='linear', inplace=True)

# Checking if any null values are left
print(df.isnull().sum())
#Feature Scaling & Encoding
from sklearn.preprocessing import StandardScaler

# Selecting all numerical columns excluding Date and time-based columns
numerical_cols = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 
                  'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
#Time based Featured Engineering
# Parsing the Date column into a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Extract time-based features
df['Hour'] = df['Date'].dt.hour
df['Day_of_Week'] = df['Date'].dt.dayofweek  
df['Month'] = df['Date'].dt.month

# Droping the original Date column
df.drop('Date', axis=1, inplace=True)

#Outlier Detection and Handling
#We are gonna be capping the values so that it does not exceed a threshold or go below one value
#Using Interquartile Range(IQR) method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers
for col in numerical_cols:
    outliers = detect_outliers(df, col)
    print(f"Outliers in {col}: {len(outliers)}")
#capping the values
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
for col in numerical_cols:
    cap_outliers(df, col)
#Exploratory Data Analysis
#Histograms
# Set up the plotting environment
plt.figure(figsize=(15, 10))
# Plot histograms for each numerical column
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 3, i)  # 4 rows, 3 columns for 11 features
    sns.histplot(data=df, x=col, kde=True)  # kde=True adds a kernel density estimate
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()
#BoxPlots
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_cols])
plt.title('Boxplots of Numerical Features')
plt.show()

#Scatter plots 
# Example: CO(GT) vs. Temperature (T)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='T', y='CO(GT)')
plt.title('CO(GT) vs. Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('CO (GT)')
plt.show()

# Example: NOx vs. Hour of the Day
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Hour', y='PT08.S3(NOx)')
plt.title('NOx vs. Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('NOx')
plt.show()