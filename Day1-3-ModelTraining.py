import pandas as pd

# Load car dataset
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

# display the first five rows
# head() method displays the first 5 rowa by default.
print(df.head())

# display the shape of the dataset
print(df.shape)

# display the data types of each column
print(df.columns)

# display the column names, data types, and non-null counts
print(df.info())

# display the number of missing values in each column
print(df.isnull().sum())

# display the number of duplicate rows in the dataset
print(df.duplicated().sum())

# display the summary statistics of the numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print(numerical_columns)

# display the summary statistics of the text-based columns
text_columns = df.select_dtypes(include=['object']).columns

print(text_columns)

