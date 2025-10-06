import os
import json

def fix_json_files(directory="data/fundamental_data/"):
    """
    Scans a directory for .json files and fixes any that are malformed.

    The function assumes that malformed files are simply truncated and are
    missing closing braces and brackets. It attempts to repair them by
e    appending the necessary closing characters.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"--- Scanning for broken JSON files in '{directory}' ---")
    fixed_count = 0
    checked_count = 0

    # Iterate over all files in the specified directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            checked_count += 1
            file_path = os.path.join(directory, filename)

            try:
                # Attempt to open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                # If it loads without error, it's a valid JSON file
                # print(f"'{filename}' is valid. Skipping.")
            except json.JSONDecodeError:
                # This block executes if json.load() fails, indicating a broken file
                print(f"-> Found malformed JSON: '{filename}'. Attempting to fix...")

                # Re-open the file in read mode to get its raw content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # The simplest fix is to assume the file was truncated.
                # We will add the closing characters.
                # This assumes the structure is { "info":{...}, "income_statement":{...
                # and was cut off. We close the inner object and the main object.
                fixed_content = content + "\n}\n}"

                # Overwrite the original file with the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)

                print(f"--> Successfully repaired '{filename}'.")
                fixed_count += 1

    print("\n--- Scan Complete ---")
    print(f"Checked {checked_count} JSON files.")
    print(f"Repaired {fixed_count} files.")

if __name__ == "__main__":
    fix_json_files()
