import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def extract_json_to_parquet(input_file_path, output_file_path):    
    extracted_data = []
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        extracted_data.append({
                            'user_id': item.get('user_id'),
                            'parent_asin': item.get('parent_asin'),
                            'rating': item.get('rating')
                        })
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")
                        continue
                
                # Progress tracker for sanity
                if line_number % 10000 == 0:
                    print(f"Processed {line_number} lines...")
                            
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{input_file_path}': {e}")
        return
    except Exception as e:
        print(f"Error reading file '{input_file_path}': {e}")
        return
    
    if not extracted_data:
        print("No data extracted from the file.")
        return
    
    # Write data to parquet
    df = pd.DataFrame(extracted_data)
    
    # Display basic info about the extracted data
    print(f"Extracted {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values to make sure we read right
    missing_values = df.isnull().sum()
    if missing_values.any():
        print(f"\nMissing values:")
        print(missing_values[missing_values > 0])
    
    # Save to parquet
    try:
        df.to_parquet(output_file_path, index=False)
        print(f"\nData successfully saved to '{output_file_path}'")
        
        # Verify the saved file
        df_verify = pd.read_parquet(output_file_path)
        print(f"Verification: Parquet file contains {len(df_verify)} records")
        
    except Exception as e:
        print(f"Error saving to parquet: {e}")

# So I can just run the script without having to copy file paths over and over
if __name__ == "__main__":
    file_directory = "/Users/matttillman/School/data_612/assignments/Final/data/"
    input_file = file_directory + "Software.jsonl"
    output_file = file_directory + "software.parquet"
    
    extract_json_to_parquet(input_file, output_file)