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
            'columns': tuple(df.columns),  # Use tuple to make it hashable
            'dtypes': tuple(df.dtypes.items()),  # Use tuple of dtype items for comparison
            'shape': df.shape
        }
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Group files by structure (columns and data types)
grouped_files = {}
for file, info in file_info.items():
    key = (info['columns'], info['dtypes'])  # Group by columns and data types
    if key not in grouped_files:
        grouped_files[key] = []
    grouped_files[key].append(file)

# Combine files in each group
output_folder = '/Users/w/Developer/250306 KYK Datamining/250314Shenfeng/combined'
os.makedirs(output_folder, exist_ok=True)

for i, (key, files) in enumerate(grouped_files.items(), 1):
    print(f"Combining files in Group {i}: {files}")
    combined_data = pd.DataFrame()
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    # Save the combined data as a new CSV file
    output_file = os.path.join(output_folder, f'group_{i}_combined.csv')
    combined_data.to_csv(output_file, index=False)
    print(f"Group {i} combined file saved as: {output_file}")