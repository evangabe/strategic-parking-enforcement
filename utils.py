import json
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

VERSION = 0
DEFAULT_FILE_PATH = f"data/pickle/violations_v{VERSION}.pickle"
LACITY_ENDPOINT_URL = "https://data.lacity.org/resource/4f5p-udkv.json"
NUM_THREADS = 10

def fetch_data(limit, offset):
    response = requests.get(f'{LACITY_ENDPOINT_URL}?$limit={limit}&$offset={offset}')
    return json.loads(response.text)

def process_chunk(chunk):
    df = pd.DataFrame(chunk)
    return transform_dataset(df)

def pickle_dataset(output_path=DEFAULT_FILE_PATH):
    violations_df_list = []
    limit = 50000
    total_rows = 0
    chunk_size = 50000
    more_data = True    

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        
        while more_data:
            current_rows = total_rows
            print(f'Rows extracted: {current_rows}')
            
            # Submit a new fetch job
            futures.append(executor.submit(fetch_data, limit, current_rows))
            
            # Process the completed futures
            if len(futures) >= NUM_THREADS:
                completed_futures = []
                for future in as_completed(futures):
                    data_chunk = future.result()
                    if data_chunk:  # Check if data_chunk is not empty
                        processed_chunk = process_chunk(data_chunk)
                        violations_df_list.append(processed_chunk)
                        total_rows += len(processed_chunk)
                        if len(data_chunk) < limit or total_rows > 21600000:
                            print("No more data available. Exiting loop.")
                            more_data = False
                            break
                    completed_futures.append(future)

                # Remove completed futures from the list
                futures = [f for f in futures if f not in completed_futures]

        print("Finished primary loop. Processing remaining futures . . .")

        # Process any remaining futures
        for future in as_completed(futures):
            data_chunk = future.result()
            if data_chunk:  # Check if data_chunk is not empty
                processed_chunk = process_chunk(data_chunk)
                violations_df_list.append(processed_chunk)
                total_rows += len(processed_chunk)
    

    print("Processing complete! Concatenating all data chunks . . .")

    # Concatenate all chunks into a single DataFrame
    final_df = pd.concat(violations_df_list, ignore_index=True)

    shape = final_df.shape
    print(f"\n\nDataFrame created with {shape[0]} rows and {shape[1]} columns.")

    final_df.dropna(subset=['issue_date', 'issue_time', 'violation_code'], inplace=True)

    print(f"{shape[0] - final_df.shape[0]} rows with missing data were dropped. \nPickling and storing data at {output_path} . . .")
    
    # Store as .pickle file
    final_df.to_pickle(output_path)

def transform_dataset(df):
    return df[['issue_date', 'issue_time', 'fine_amount', 'agency', 'violation_code', 'loc_lat', 'loc_long']]

def get_dataset_from_pickle(input_path=DEFAULT_FILE_PATH):
    """
    
    """
    return pd.read_pickle(input_path)

