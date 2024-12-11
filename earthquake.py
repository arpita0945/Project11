import pandas as pd #handle data
import seaborn as sns #create boxplot n countplot
import matplotlib.pyplot as plt #display plot

#after importing library we need the dataset which we downloaded from internet

# Step 1: Load the dataset
df = pd.read_csv('C:\\Users\\kumar\\OneDrive\\Documents\\earthquake_1995-2023.csv') #we have given path for reading the dataset
print(df)

# Step 2: Display the first few rows of the dataset
print("First 5 rows of the dataset:") #after reading the dataset we display first five rows of the dataset
print(df.head())

# Step 3: Clean the dataset
def clean_dataset(df): #we have define function clean dataset and parse argument df
    # 1. Drop rows with missing values in crucial columns (e.g., 'magnitude', 'depth')
    df = df.dropna(subset=['magnitude', 'depth'])#we have deleted the row with magnitude and depth with the value Nan
    
    # 2. Remove duplicate rows
    df = df.drop_duplicates()#we have remove duplicate rows for accuracy
    
    # 3. Convert necessary columns to appropriate data types (if needed)
    # For example, converting 'magnitude' and 'depth' to numeric if they are not already
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    
    # Check for remaining missing values
    print("\nMissing Values After Cleaning:")#it will check if any missing value
    print(df.isnull().sum())
    
    # Final shape of the dataset
    print(f"\nDataframe shape after cleaning: {df.shape}")
    return df
# Clean the dataset
df_cleaned = clean_dataset(df)#here,we are displaying the clean dataset

# Step 4: Basic statistics 
def basic_statistics(df): #function name and it take table of earthquake data
    print("\n== Basic Earthquake Statistics ==") #it print title
    print(f"Total number of earthquakes: {len(df)}") # count how many earthquakes r in table n show that no.
    print(f"Maximum magnitude: {df['magnitude'].max()}")#finds the largest eartquake n show its size
    print(f"Minimum magnitude: {df['magnitude'].min()}")#finds the smallest eathquake n show its size
    print(f"Mean magnitude: {df['magnitude'].mean()}")#calculate the average size of earthquakes 
    print(f"Standard deviation of magnitude: {df['magnitude'].std()}")#how much sizes vary
    print(f"Total number of unique locations: {df['location'].nunique() if 'location' in df.columns else 'N/A'}")#how many diff. places had eartquakes

# Print basic statistics of the cleaned dataset
basic_statistics(df_cleaned) #summarize the data

# Step 5: Visualization

# 5.1: Histogram of Earthquake Magnitudes
plt.figure(figsize=(10, 6)) #creates a new figure for plot 
sns.histplot(df_cleaned['magnitude'], bins=30, kde=True)#sns.histplot is function of seaborn to plot histogram chart
plt.title('Distribution of Earthquake Magnitudes')#we have adding the title of plot
plt.xlabel('Magnitude')#name of x-axis 
plt.ylabel('Frequency')#and y-axis
plt.grid(True)#for better readability of plot
plt.show()

# 5.2: Scatter plot of Magnitude vs Depth
#creating scatter plot with depth on x-axis and magnitude on y-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='depth', y='magnitude', data=df_cleaned)
plt.title('Magnitude vs Depth of Earthquakes')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# 5.3: Correlation matrix and heatmap
# Select numeric columns for correlation analysis
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])#used to select only numeric column that is numerical data 

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)#sns.heatmap is function of seaboorn to plot 
plt.show()

# Step 6: Analyze largest earthquake
largest_earthquake = df_cleaned[df_cleaned['magnitude'] == df_cleaned['magnitude'].max()]#how many diff. places had eartquakes
print("\nLargest Earthquake Information:")#individual value in different colour for east to understand
plt.title('Correlation Matrix for Numeric Columns')
plt.grid(True)
print(largest_earthquake)


# Step 7: Visualization of Largest Earthquake
#with the help of scatterplot we are showing largest earthquake
plt.figure(figsize=(10, 6))#creates a new figure for plot 
sns.scatterplot(x='depth', y='magnitude', data=df_cleaned, label='All Earthquakes', alpha=0.5)
plt.scatter(largest_earthquake['depth'], largest_earthquake['magnitude'], color='red', s=200, label='Largest Earthquake', zorder=5)
plt.title('Largest Earthquake in the Dataset')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)#for better readability of plot
plt.show()





