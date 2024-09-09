import json
import os
import argparse
import shutil

def check_and_fix_json_files(directory, fix=False):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and not file.startswith('broken_'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    if "Extra data" in str(e):
                        with open(file_path, 'r') as f:
                            content = f.read().strip()
                        
                        if content.endswith('}') and content.count('}') > content.count('{'):
                            print(f"File with trailing '}}': {file_path}")
                            
                            if fix:
                                # Create a backup
                                backup_path = os.path.join(root, f"broken_{file}")
                                shutil.copy2(file_path, backup_path)
                                
                                # Fix the file
                                fixed_content = content.rstrip('}')
                                with open(file_path, 'w') as f:
                                    f.write(fixed_content)
                                print(f"  Fixed and backup created: {backup_path}")
                            else:
                                print("  Use --fix=y to fix this file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and optionally fix JSON files with trailing '}'")
    parser.add_argument("--fix", choices=['y', 'n'], default='n', help="Fix the broken JSON files (y/n)")
    args = parser.parse_args()

    fix = args.fix == 'y'
    check_and_fix_json_files('./saves', fix)
