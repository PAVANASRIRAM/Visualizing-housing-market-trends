import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the housing dataset
# Replace 'housing.csv' with your actual CSV file path
df = pd.read_csv('housing.csv')

# Display basic info about the dataset
print("Dataset Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Plot: Correlation matrix heatmap between features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Housing Features')
plt.tight_layout()
plt.show()

# Plot: Distribution of Sale Prices
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.tight_layout()
plt.show()

# Plot: Sale Price vs. Square Feet, colored by Bedrooms
if 'SquareFeet' in df.columns and 'Bedrooms' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='SquareFeet', y='SalePrice', hue='Bedrooms', palette='viridis')
    plt.title('Sale Price vs. Square Feet (Colored by Bedrooms)')
    plt.xlabel('Square Feet')
    plt.ylabel('Sale Price')
    plt.tight_layout()
    plt.show()

# Plot: Boxplot of Sale Price by Number of Bedrooms
if 'Bedrooms' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Bedrooms', y='SalePrice')
    plt.title('Sale Price by Number of Bedrooms')
    plt.xlabel('Bedrooms')
    plt.ylabel('Sale Price')
    plt.tight_layout()
    plt.show()

# Plot: Average Sale Price by Year Built (trend over time)
if 'YearBuilt' in df.columns:
    df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')
    avg_price_by_year = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_price_by_year, x='YearBuilt', y='SalePrice')
    plt.title('Average Sale Price by Year Built')
    plt.xlabel('Year Built')
    plt.ylabel('Average Sale Price')
    plt.tight_layout()
    plt.show()
