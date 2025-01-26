# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset into dataframe
df = pd.read_csv('Carseats.csv')

# Displaying basic information about the dataset
print("Shape of the dataset:", df.shape)
print("\nDataset information:\n")
df.info()
print("\nDescriptive statistics:\n")
print(df.describe())

# Handling missing values by replacing with median
df['Sales'].fillna(df['Sales'].median(), inplace=True)
df['CompPrice'].fillna(df['CompPrice'].median(), inplace=True)
df['Income'].fillna(df['Income'].median(), inplace=True)
df['Urban'].fillna(df['Urban'].mode()[0], inplace=True)

# Converting float values to integers
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype(int)

# Removing duplicate entries
df.drop_duplicates(inplace=True)

# Customizing plot aesthetics
custom_palette = sns.color_palette("Set2")
sns.set(style="whitegrid")

# Histogram of Sales
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], bins=20, kde=True, color=custom_palette[0])
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Sales vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Sales', data=df, hue='ShelveLoc', palette=custom_palette)
plt.title('Sales vs. Price by Shelve Location')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.legend(title='ShelveLoc', loc='upper right')
plt.show()

# Box plot of Sales by Shelve Location
plt.figure(figsize=(10, 6))
sns.boxplot(x='ShelveLoc', y='Sales', data=df, palette=custom_palette)
plt.title('Sales by Shelve Location')
plt.xlabel('Shelve Location')
plt.ylabel('Sales')
plt.show()

# Relationship between Sales, Income, and Price by Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='Sales', hue='Price', data=df, palette='viridis')
plt.title('Sales vs. Income by Price')
plt.xlabel('Income')
plt.ylabel('Sales')
plt.legend(title='Price')
plt.show()

# Unique values of Shelve Location and their frequencies
shelveloc_counts = df['ShelveLoc'].value_counts()
print("\nUnique values and their frequencies:\n")
print(shelveloc_counts)

# Contingency table and chi-square test between Urban and US
cont_table = pd.crosstab(df['Urban'], df['US'])
chi2_stat, p_val, _, _ = stats.chi2_contingency(cont_table)
print("\nContingency Table between Urban and US:\n")
print(cont_table)
print(f"\nChi-square test result: chi2 = {chi2_stat:.2f}, p-value = {p_val:.4f}")

# Subset of rows based on criteria and descriptive statistics
median_sales = df['Sales'].median()
subset_data = df[(df['Urban'] == 'Yes') & (df['Sales'] > median_sales)]
print("\nDescriptive statistics for subset (Urban='Yes' and Sales > median):\n")
print(subset_data.describe())

# T-test for Sales between Good and Bad Shelve Location
sales_good = df[df['ShelveLoc'] == 'Good']['Sales']
sales_bad = df[df['ShelveLoc'] == 'Bad']['Sales']
t_stat, p_val = stats.ttest_ind(sales_good, sales_bad, equal_var=False)
print(f"\nT-test result for Sales between ShelveLoc 'Good' and 'Bad': t-stat = {t_stat:.2f}, p-value = {p_val:.4f}")

# Grouped summary statistics of Mean Sales by ShelveLoc
mean_sales_shelveloc = df.groupby('ShelveLoc')['Sales'].mean().reset_index()
print("\nMean Sales by ShelveLoc:")
print(mean_sales_shelveloc)

# Linear Regression Model
# Converting categorical columns to dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Defining features and target variable
X = df_encoded.drop('Sales', axis=1)
y = df_encoded['Sales']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating and printing the accuracy metrics in a table format
accuracy_metrics = pd.DataFrame({
    'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R2)'],
    'Value': [mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)]
})
print("\nAccuracy Metrics:\n")
print(accuracy_metrics)
