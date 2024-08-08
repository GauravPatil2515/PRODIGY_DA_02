import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Print the directory name (corrected path handling)
print("Directory:", os.path.basename(r'C:\Users\GAURAV PATIL\Downloads'))

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(df.head())

# Check the dimensions of the dataset
print("Shape of dataset:", df.shape)

# Get summary statistics of numerical variables
print("Summary statistics:")
print(df.describe())

# Check the data types of variables
print("Data types:")
print(df.dtypes)

# Check for missing values
print("Missing values count:")
print(df.isnull().sum())

# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Bar plot of survival count
sns.countplot(x='survived', data=df)
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.title('Survival Count')
plt.show()

# Histogram of age distribution
plt.hist(df['age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Scatter plot of Age vs. Fare
plt.scatter(df['age'], df['fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs. Fare')
plt.show()

# Box plot of Fare by Survival Status
sns.boxplot(x='survived', y='fare', data=df)
plt.xlabel('Survival Status')
plt.ylabel('Fare')
plt.title('Survival Status vs. Fare')
plt.show()

# Correlation analysis
correlation = df[['age', 'fare']].corr()
print("Correlation between Age and Fare:")
print(correlation)

# Cross-tabulation of Pclass and Survival
cross_tab = pd.crosstab(df['pclass'], df['survived'])
print("Cross-tabulation of Pclass and Survival:")
print(cross_tab)
