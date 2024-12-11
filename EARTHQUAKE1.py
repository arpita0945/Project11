import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('C:\\Users\\kumar\\OneDrive\\Documents\\earthquake_1995-2023.csv')
print(df)

# Step 2: Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Clean the dataset
def clean_dataset(df):
    # 1. Drop rows with missing values in crucial columns (e.g., 'magnitude', 'depth')
    df = df.dropna(subset=['magnitude', 'depth'])
    
    # 2. Remove duplicate rows
    df = df.drop_duplicates()
    
    # 3. Convert necessary columns to appropriate data types (if needed)
    # For example, converting 'magnitude' and 'depth' to numeric if they are not already
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    
    # Check for remaining missing values
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())
    
    # Final shape of the dataset
    print(f"\nDataframe shape after cleaning: {df.shape}")
    return df

# Clean the dataset
df_cleaned = clean_dataset(df)

# Step 4: Basic statistics
def basic_statistics(df):
    print("\n== Basic Earthquake Statistics ==")
    print(f"Total number of earthquakes: {len(df)}")
    print(f"Maximum magnitude: {df['magnitude'].max()}")
    print(f"Minimum magnitude: {df['magnitude'].min()}")
    print(f"Mean magnitude: {df['magnitude'].mean()}")
    print(f"Standard deviation of magnitude: {df['magnitude'].std()}")
    print(f"Total number of unique locations: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")

# Print basic statistics of the cleaned dataset
basic_statistics(df_cleaned)

# Step 5: Visualization

# 5.1: Histogram of Earthquake Magnitudes
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['magnitude'], bins=30, kde=True)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 5.2: Scatter plot of Magnitude vs Depth
plt.figure(figsize=(10, 6))
sns.scatterplot(x='depth', y='magnitude', data=df_cleaned)
plt.title('Magnitude vs Depth of Earthquakes')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


# 5.3: Correlation matrix and heatmap
# Select numeric columns for correlation analysis
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Numeric Columns')
plt.grid(True)
plt.show()

# Step 6: Analyze largest earthquake
largest_earthquake = df_cleaned[df_cleaned['magnitude'] == df_cleaned['magnitude'].max()]
print("\nLargest Earthquake Information:")
print(largest_earthquake)


# Step 7: Visualization of Largest Earthquake
plt.figure(figsize=(10, 6))
sns.scatterplot(x='depth', y='magnitude', data=df_cleaned, label='All Earthquakes', alpha=0.5)
plt.scatter(largest_earthquake['depth'], largest_earthquake['magnitude'], color='red', s=200, label='Largest Earthquake', zorder=5)
plt.title('Largest Earthquake in the Dataset')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()


# Step 8: Summary Statistics for Numerical Columns
print("\nSummary Statistics for Numerical Columns:")
print(df_cleaned.describe())