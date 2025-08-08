# %%
import pandas as pd
import numpy as np

# %%
# Creating DataFrames and Series
# Series - 1D labeled array
series1 = pd.Series([1, 2, 3, 4, 5])
series2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
series3 = pd.Series({'apple': 3, 'banana': 2, 'orange': 5})
print(f"Series 1:\n{series1}\n")
print(f"Series 2:\n{series2}\n")
print(f"Series 3:\n{series3}\n")

# DataFrame - 2D labeled data structure
# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)
print(f"DataFrame from dictionary:\n{df}\n")

# From list of lists
data_list = [['Alice', 25, 'New York'], ['Bob', 30, 'London'], ['Charlie', 35, 'Tokyo']]
df_list = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])
print(f"DataFrame from list:\n{df_list}\n")

# From numpy array
np_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df_numpy = pd.DataFrame(np_data, columns=['A', 'B', 'C'])
print(f"DataFrame from numpy array:\n{df_numpy}\n")

# %%
# DataFrame Properties and Info
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame size: {df.size}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"DataFrame index: {df.index.tolist()}")
print(f"DataFrame data types:\n{df.dtypes}\n")
print(f"DataFrame info:")
df.info()
print(f"\nDataFrame description:\n{df.describe()}\n")

# %%
# Data Selection and Indexing
# Column selection
print(f"Select single column:\n{df['Name']}\n")
print(f"Select multiple columns:\n{df[['Name', 'Age']]}\n")

# Row selection using loc (label-based) and iloc (position-based)
print(f"Select row by index using loc:\n{df.loc[0]}\n")
print(f"Select multiple rows using loc:\n{df.loc[0:2]}\n")
print(f"Select row by position using iloc:\n{df.iloc[0]}\n")
print(f"Select specific rows and columns:\n{df.loc[0:2, 'Name':'City']}\n")

# Boolean indexing
print(f"Rows where Age > 28:\n{df[df['Age'] > 28]}\n")
print(f"Rows where City is 'London' or 'Paris':\n{df[df['City'].isin(['London', 'Paris'])]}\n")

# Query method
print(f"Using query method:\n{df.query('Age > 28 and Salary >= 60000')}\n")

# %%
# Adding and Modifying Data
# Add new column
df['Bonus'] = df['Salary'] * 0.1
print(f"DataFrame with Bonus column:\n{df}\n")

# Add column based on condition
df['Senior'] = df['Age'] >= 30
print(f"DataFrame with Senior column:\n{df}\n")

# Modify existing column
df['Salary'] = df['Salary'] * 1.05  # 5% raise
print(f"DataFrame after salary increase:\n{df}\n")

# Add new row
new_row = {'Name': 'Eve', 'Age': 32, 'City': 'Berlin', 'Salary': 65000, 'Bonus': 6500, 'Senior': True}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print(f"DataFrame after adding new row:\n{df}\n")

# %%
# Data Cleaning and Handling Missing Values
# Create DataFrame with missing values
data_with_nan = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, np.nan, 5],
    'D': ['x', 'y', np.nan, 'z', 'w']
}
df_nan = pd.DataFrame(data_with_nan)
print(f"DataFrame with missing values:\n{df_nan}\n")

# Check for missing values
print(f"Missing values per column:\n{df_nan.isnull().sum()}\n")
print(f"Missing values (boolean):\n{df_nan.isnull()}\n")

# Drop missing values
print(f"Drop rows with any NaN:\n{df_nan.dropna()}\n")
print(f"Drop columns with any NaN:\n{df_nan.dropna(axis=1)}\n")
print(f"Drop rows only if all values are NaN:\n{df_nan.dropna(how='all')}\n")

# Fill missing values
print(f"Fill NaN with 0:\n{df_nan.fillna(0)}\n")
print(f"Forward fill:\n{df_nan.ffill()}\n")
print(f"Backward fill:\n{df_nan.bfill()}\n")
print(f"Fill with mean (numeric columns only):\n{df_nan.fillna(df_nan.mean(numeric_only=True))}\n")

# %%
# Data Types and Conversion
print(f"Original data types:\n{df.dtypes}\n")

# Convert data types
df_copy = df.copy()
df_copy['Age'] = df_copy['Age'].astype('float64')
df_copy['Senior'] = df_copy['Senior'].astype('str')
print(f"After type conversion:\n{df_copy.dtypes}\n")

# String operations
df_copy['Name_Upper'] = df_copy['Name'].str.upper()
df_copy['Name_Length'] = df_copy['Name'].str.len()
print(f"DataFrame with string operations:\n{df_copy[['Name', 'Name_Upper', 'Name_Length']].head()}\n")

# %%
# Grouping and Aggregation
# Create sample data for grouping
sales_data = {
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
    'Product': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 120, 110, 180, 190, 140],
    'Profit': [20, 30, 40, 25, 22, 36, 38, 28]
}
sales_df = pd.DataFrame(sales_data)
print(f"Sales DataFrame:\n{sales_df}\n")

# Group by single column
print(f"Group by Region - sum:\n{sales_df.groupby('Region').sum()}\n")
print(f"Group by Region - mean:\n{sales_df.groupby('Region').mean(numeric_only=True)}\n")

# Group by multiple columns
print(f"Group by Region and Product:\n{sales_df.groupby(['Region', 'Product']).sum()}\n")

# Multiple aggregations
print(f"Multiple aggregations:\n{sales_df.groupby('Region').agg({'Sales': ['sum', 'mean'], 'Profit': ['sum', 'max']})}\n")

# Custom aggregation
def profit_margin(group):
    return (group['Profit'].sum() / group['Sales'].sum()) * 100

print(f"Custom aggregation (profit margin):\n{sales_df.groupby('Region').apply(profit_margin)}\n")

# %%
# Sorting and Ranking
print(f"Sort by Age (ascending):\n{df.sort_values('Age')}\n")
print(f"Sort by multiple columns:\n{df.sort_values(['Age', 'Salary'], ascending=[True, False])}\n")
print(f"Sort by index:\n{df.sort_index()}\n")

# Ranking
df['Age_Rank'] = df['Age'].rank()
df['Salary_Rank'] = df['Salary'].rank(ascending=False)
print(f"DataFrame with rankings:\n{df[['Name', 'Age', 'Age_Rank', 'Salary', 'Salary_Rank']]}\n")

# %%
# Merging and Joining DataFrames
# Create sample DataFrames for merging
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Department': ['HR', 'IT', 'Finance', 'IT']
})

df2 = pd.DataFrame({
    'ID': [2, 3, 4, 5],
    'Salary': [60000, 70000, 55000, 65000],
    'Bonus': [6000, 7000, 5500, 6500]
})

print(f"DataFrame 1:\n{df1}\n")
print(f"DataFrame 2:\n{df2}\n")

# Different types of joins
print(f"Inner join:\n{pd.merge(df1, df2, on='ID', how='inner')}\n")
print(f"Left join:\n{pd.merge(df1, df2, on='ID', how='left')}\n")
print(f"Right join:\n{pd.merge(df1, df2, on='ID', how='right')}\n")
print(f"Outer join:\n{pd.merge(df1, df2, on='ID', how='outer')}\n")

# Concatenation
df3 = pd.DataFrame({
    'ID': [6, 7],
    'Name': ['Frank', 'Grace'],
    'Department': ['Marketing', 'Sales']
})

print(f"Concatenate vertically:\n{pd.concat([df1, df3], ignore_index=True)}\n")
print(f"Concatenate horizontally:\n{pd.concat([df1, df2], axis=1)}\n")

# %%
# Pivot Tables and Cross-tabulation
# Create sample data for pivot table
pivot_data = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'Sales': [100, 150, 110, 160, 120, 170]
}
pivot_df = pd.DataFrame(pivot_data)
print(f"Pivot data:\n{pivot_df}\n")

# Create pivot table
pivot_table = pd.pivot_table(pivot_df, values='Sales', index='Date', columns='Product', aggfunc='sum')
print(f"Pivot table:\n{pivot_table}\n")

# Cross-tabulation
cross_tab = pd.crosstab(pivot_df['Product'], pivot_df['Region'], values=pivot_df['Sales'], aggfunc='sum')
print(f"Cross-tabulation:\n{cross_tab}\n")

# %%
# Date and Time Operations
# Create DataFrame with datetime
date_data = {
    'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
    'Value': np.random.randn(10)
}
date_df = pd.DataFrame(date_data)
print(f"DataFrame with dates:\n{date_df}\n")

# Extract date components
date_df['Year'] = date_df['Date'].dt.year
date_df['Month'] = date_df['Date'].dt.month
date_df['DayOfWeek'] = date_df['Date'].dt.dayofweek
date_df['WeekdayName'] = date_df['Date'].dt.day_name()
print(f"DataFrame with date components:\n{date_df}\n")

# Set date as index for time series operations
date_df.set_index('Date', inplace=True)
print(f"DataFrame with date index:\n{date_df}\n")

# Resample (time series aggregation)
print(f"Weekly resample (mean):\n{date_df['Value'].resample('W').mean()}\n")

# %%
# String Operations
text_data = {
    'Text': ['Hello World', 'pandas is GREAT', 'Data Science', 'Machine Learning', 'AI and ML']
}
text_df = pd.DataFrame(text_data)

# String methods
text_df['Lower'] = text_df['Text'].str.lower()
text_df['Upper'] = text_df['Text'].str.upper()
text_df['Length'] = text_df['Text'].str.len()
text_df['Word_Count'] = text_df['Text'].str.split().str.len()
text_df['Contains_Data'] = text_df['Text'].str.contains('Data', case=False)
text_df['First_Word'] = text_df['Text'].str.split().str[0]
print(f"String operations:\n{text_df}\n")

# Regular expressions
text_df['Extract_Uppercase'] = text_df['Text'].str.extract(r'([A-Z]+)')
print(f"Extract uppercase letters:\n{text_df[['Text', 'Extract_Uppercase']]}\n")

# %%
# Advanced Indexing and MultiIndex
# Create MultiIndex DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
multi_df = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['X', 'Y'])
print(f"MultiIndex DataFrame:\n{multi_df}\n")

# Access MultiIndex data
print(f"Select level A:\n{multi_df.loc['A']}\n")
print(f"Select specific combination:\n{multi_df.loc[('A', 'one')]}\n")

# Reset and set index
print(f"Reset index:\n{multi_df.reset_index()}\n")

# %%
# Data Transformation and Apply Functions
# Apply functions to DataFrame
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 40:
        return 'Middle'
    else:
        return 'Senior'

# Using our original df
df['Age_Category'] = df['Age'].apply(categorize_age)
print(f"DataFrame with age categories:\n{df[['Name', 'Age', 'Age_Category']]}\n")

# Apply to multiple columns
def calculate_tax(row):
    return row['Salary'] * 0.2 if row['Senior'] else row['Salary'] * 0.15

df['Tax'] = df.apply(calculate_tax, axis=1)
print(f"DataFrame with tax calculation:\n{df[['Name', 'Salary', 'Senior', 'Tax']].head()}\n")

# Map function
city_mapping = {'New York': 'USA', 'London': 'UK', 'Tokyo': 'Japan', 'Paris': 'France', 'Berlin': 'Germany'}
df['Country'] = df['City'].map(city_mapping)
print(f"DataFrame with country mapping:\n{df[['Name', 'City', 'Country']].head()}\n")

# %%
# Statistical Operations
stats_data = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(5, 2, 100),
    'C': np.random.exponential(2, 100)
})

print(f"Basic statistics:\n{stats_data.describe()}\n")
print(f"Correlation matrix:\n{stats_data.corr()}\n")
print(f"Covariance matrix:\n{stats_data.cov()}\n")

# Rolling statistics
stats_data['A_Rolling_Mean'] = stats_data['A'].rolling(window=10).mean()
stats_data['B_Rolling_Std'] = stats_data['B'].rolling(window=10).std()
print(f"First 15 rows with rolling statistics:\n{stats_data[['A', 'A_Rolling_Mean', 'B', 'B_Rolling_Std']].head(15)}\n")

# %%
# Working with Categorical Data
# Create categorical data
categories = ['Low', 'Medium', 'High']
cat_data = pd.Categorical(['Low', 'High', 'Medium', 'Low', 'High'], categories=categories, ordered=True)
cat_df = pd.DataFrame({'Category': cat_data, 'Value': [10, 30, 20, 15, 35]})
print(f"Categorical DataFrame:\n{cat_df}\n")
print(f"Category info:\n{cat_df['Category'].cat.categories}\n")

# Sort by categorical order
print(f"Sorted by category:\n{cat_df.sort_values('Category')}\n")

# %%
# Performance Tips and Memory Optimization
# Create large DataFrame for demonstration
large_df = pd.DataFrame({
    'A': np.random.randint(0, 100, 10000),
    'B': np.random.random(10000),
    'C': ['category_' + str(i % 10) for i in range(10000)]
})

print(f"Memory usage before optimization:\n{large_df.memory_usage(deep=True)}\n")

# Optimize data types
large_df['A'] = large_df['A'].astype('int8')  # Since values are 0-99
large_df['C'] = large_df['C'].astype('category')  # Convert to categorical

print(f"Memory usage after optimization:\n{large_df.memory_usage(deep=True)}\n")

# %%
# File I/O Operations (commented out to avoid file dependencies)
# Reading and writing files
"""
# Read CSV
df_csv = pd.read_csv('data.csv')

# Write CSV
df.to_csv('output.csv', index=False)

# Read Excel
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Write Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Read JSON
df_json = pd.read_json('data.json')

# Write JSON
df.to_json('output.json', orient='records')

# Read from SQL (requires SQLAlchemy)
# df_sql = pd.read_sql('SELECT * FROM table_name', connection)
"""

# %%
# Common Interview Questions and Solutions

# 1. Find duplicate rows
duplicate_data = pd.DataFrame({
    'A': [1, 2, 2, 3, 4],
    'B': [5, 6, 6, 7, 8],
    'C': [9, 10, 10, 11, 12]
})
print(f"Original data:\n{duplicate_data}\n")
print(f"Duplicate rows:\n{duplicate_data[duplicate_data.duplicated()]}\n")
print(f"Remove duplicates:\n{duplicate_data.drop_duplicates()}\n")

# 2. Find second highest value
print(f"Second highest salary:\n{df['Salary'].nlargest(2).iloc[-1]}\n")

# 3. Percentage of missing values
def missing_percentage(df):
    return (df.isnull().sum() / len(df)) * 100
print(f"Missing percentage in df_nan:\n{missing_percentage(df_nan)}\n")

# 4. Moving average
def moving_average_pandas(series, window):
    return series.rolling(window=window).mean()

sample_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Moving average (window=3):\n{moving_average_pandas(sample_series, 3)}\n")

# 5. Cumulative sum and percentage
df['Cumulative_Salary'] = df['Salary'].cumsum()
df['Salary_Percentage'] = (df['Salary'] / df['Salary'].sum()) * 100
print(f"Cumulative and percentage calculations:\n{df[['Name', 'Salary', 'Cumulative_Salary', 'Salary_Percentage']].head()}\n")

# 6. Binning/Discretization
df['Age_Bins'] = pd.cut(df['Age'], bins=[0, 25, 35, 50], labels=['Young', 'Middle', 'Senior'])
print(f"Age binning:\n{df[['Name', 'Age', 'Age_Bins']].head()}\n")

# 7. Rank within groups
df['Salary_Rank_by_City'] = df.groupby('City')['Salary'].rank(ascending=False)
print(f"Salary rank within each city:\n{df[['Name', 'City', 'Salary', 'Salary_Rank_by_City']].head()}\n")

# 8. Calculate percentage change
time_series = pd.Series([100, 105, 110, 108, 115, 120])
pct_change = time_series.pct_change()
print(f"Percentage change:\n{pct_change}\n")

# 9. Conditional replacement
df['High_Earner'] = np.where(df['Salary'] > df['Salary'].median(), 'High', 'Low')
print(f"High earner classification:\n{df[['Name', 'Salary', 'High_Earner']].head()}\n")

# 10. Data profiling function
def profile_dataframe(df):
    profile = pd.DataFrame({
        'Column': df.columns,
        'Non_Null_Count': df.count(),
        'Null_Count': df.isnull().sum(),
        'Null_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Dtype': df.dtypes,
        'Unique_Values': df.nunique()
    })
    return profile

print(f"Data profiling:\n{profile_dataframe(df)}\n")

# %%