import os
import pandas as pd

# Folder containing the CSV files
folder_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/datasets'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Dictionary to store information about each file
file_info = {}

# Loop through each CSV file and collect information
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path)
        file_info[file] = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape
        }
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Compare the files
print("Comparing CSV files...\n")
for file, info in file_info.items():
    print(f"File: {file}")
    print(f"Columns: {info['columns']}")
    print(f"Data Types: {info['dtypes']}")
    print(f"Shape: {info['shape']}")
    print("-" * 50)

# Check if all files have the same format
columns_set = {tuple(info['columns']) for info in file_info.values()}
if len(columns_set) == 1:
    print("All files have the same columns.")
else:
    print("Files have different columns.")

# Optional: Check for differences in data types
dtypes_set = {tuple(info['dtypes'].items()) for info in file_info.values()}
if len(dtypes_set) == 1:
    print("All files have the same data types.")
else:
    print("Files have different data types.")
    
    
    
    """
    
    import pandas as pd

# File paths for the two CSV files
file1_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/2023-open-data-dfb-ambulance.csv'
file2_path = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/another-file.csv'  # Replace with the second file's name

try:
    # Load the two CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Compare column names
    print("Columns in File 1:", df1.columns)
    print("Columns in File 2:", df2.columns)
    print("\nDo the files have the same columns?", list(df1.columns) == list(df2.columns))
    
    # Compare data types
    print("\nData types in File 1:")
    print(df1.dtypes)
    print("\nData types in File 2:")
    print(df2.dtypes)
    print("\nDo the files have the same data types?", df1.dtypes.equals(df2.dtypes))
    
    # Compare basic statistics
    print("\nBasic statistics for File 1:")
    print(df1.describe())
    print("\nBasic statistics for File 2:")
    print(df2.describe())
    
except Exception as e:
    print(f"An error occurred: {e}")
    """