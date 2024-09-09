import os
import json

def process_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and not file.startswith('broken_'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    #print(f"Successfully loaded: {file_path}")
                except json.JSONDecodeError:
                    new_name = f"broken_{file}"
                    new_path = os.path.join(root, new_name)
                    os.rename(file_path, new_path)
                    print(f"Renamed {file_path} to {new_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

# Specify the directory path
saves_directory = "./saves"

# Call the function to process JSON files
process_json_files(saves_directory)
