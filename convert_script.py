import pandas as pd
import re
import os
import json
import traceback # Import traceback

# --- Configuration ---
TS_FILE_PATH = 'twi_speech_app/constants/script.ts'
EXCEL_FILE_PATH = 'twi_speech_app/scripts/recording_script.xlsx'
OUTPUT_TS_FILE_PATH ='twi_speech_app/constants/script_from_excel.ts' # Avoid overwriting original
VARIABLE_NAME = 'RECORDING_SCRIPT'
TYPE_IMPORT_LINE = "import { ScriptPrompt } from '@/types';" # Adjust if needed
TYPE_ANNOTATION = "ScriptPrompt[]"

# --- Helper Functions ---
# escape_ts_string and parse_ts_array_from_file remain the same
def escape_ts_string(value):
    """Escapes single quotes and backslashes for TypeScript single-quoted strings."""
    if not isinstance(value, str):
        value = str(value)
    return value.replace('\\', '\\\\').replace("'", "\\'")

def parse_ts_array_from_file(filepath, variable_name):
    """Parses the array content from a TypeScript file."""
    # ... (keep the existing implementation from the previous good version) ...
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: TypeScript file not found at {filepath}")
        return None
    pattern = re.compile(
        r"export\s+const\s+" + re.escape(variable_name) + r"\s*:\s*[\w\<\>\[\]]+\s*=\s*(\[(?:.|\n)*?\])\s*;",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(content)
    if not match:
        print(f"Error: Could not find the variable '{variable_name}' array structure in {filepath}")
        return None
    array_content = match.group(1)
    array_content = array_content.strip()[1:-1].strip()
    object_pattern = re.compile(
        r"\{\s*"
        r"id:\s*'(?P<id>(?:\\.|[^'])*)',\s*"
        r"type:\s*'(?P<type>(?:\\.|[^'])*)',\s*"
        r"text:\s*'(?P<text>(?:\\.|[^'])*)'"
        r"(?:,\s*meaning:\s*'(?P<meaning>(?:\\.|[^'])*)')?"
        r"\s*\}",
        re.DOTALL | re.IGNORECASE
    )
    data = []
    for obj_match in object_pattern.finditer(array_content):
        entry = obj_match.groupdict()
        for key, value in entry.items():
            if value is not None:
                 entry[key] = value.replace("\\'", "'").replace("\\\\", "\\")
            elif key == 'meaning':
                 entry[key] = None
        if 'meaning' not in entry:
            entry['meaning'] = None
        data.append(entry)
    if not data and array_content:
         print(f"Warning: Array content found for '{variable_name}' but no objects were extracted.")
    elif not data:
         print(f"Warning: No objects extracted from the array '{variable_name}' in {filepath}.")
    return data

def ts_to_excel(ts_filepath, excel_filepath, variable_name):
    """Reads the TS file, parses the array, and saves it to Excel."""
    # ... (keep the existing implementation from the previous good version) ...
    print(f"Reading TypeScript data from: {ts_filepath}")
    data = parse_ts_array_from_file(ts_filepath, variable_name)
    if data is None:
        print("Aborting TS to Excel conversion.")
        return
    if not data:
        print("No data parsed from TypeScript file. Cannot create Excel file.")
        return
    print(f"Parsed {len(data)} entries.")
    df = pd.DataFrame(data)
    cols = ['id', 'type', 'text', 'meaning']
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols]
    try:
        os.makedirs(os.path.dirname(excel_filepath), exist_ok=True)
        print(f"Saving data to Excel file: {excel_filepath}")
        df.to_excel(excel_filepath, index=False, engine='openpyxl')
        print("Excel file saved successfully.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")


def excel_to_ts(excel_filepath, ts_filepath, variable_name, type_import, type_annotation):
    """Reads an Excel file and generates a TypeScript file with the array."""
    df = None
    TEMP_NULL_PLACEHOLDER = "__WAS_NULL__" # Define a unique placeholder

    try:
        print(f"Reading data from Excel file: {excel_filepath}")
        print("STEP 1: Attempting pd.read_excel...")
        # Keep na_values to convert blanks etc. in Excel to NaN
        df = pd.read_excel(excel_filepath, engine='openpyxl', na_values=['', 'NA', 'N/A', '#N/A'])
        print("STEP 1: pd.read_excel COMPLETED.")
        print("DataFrame info immediately after read_excel:")
        df.info()

        # --- Fill NaN values ---
        print(f"STEP 2: Attempting df['meaning'].fillna with placeholder '{TEMP_NULL_PLACEHOLDER}'...")
        # *** WORKAROUND: Fill 'meaning' NaNs with a string placeholder first ***
        df['meaning'] = df['meaning'].fillna(value=TEMP_NULL_PLACEHOLDER)
        print("STEP 2: COMPLETED.")

        print("STEP 3: Attempting df['id'].fillna(value='')...")
        df['id'] = df['id'].fillna(value='')
        print("STEP 3: COMPLETED.")

        print("STEP 4: Attempting df['type'].fillna(value='')...")
        df['type'] = df['type'].fillna(value='')
        print("STEP 4: COMPLETED.")

        print("STEP 5: Attempting df['text'].fillna(value='')...")
        df['text'] = df['text'].fillna(value='')
        print("STEP 5: COMPLETED.")
        # --- End of fill NaN ---

        # Ensure id is treated as string
        print("STEP 6: Attempting df['id'].astype(str)...")
        df['id'] = df['id'].astype(str)
        print("STEP 6: COMPLETED.")

        print(f"Read {len(df)} entries from Excel.")

        # --- Now, replace the placeholder with actual None ---
        # This uses standard Python comparison, not pandas functions
        print(f"STEP 7: Replacing '{TEMP_NULL_PLACEHOLDER}' with None in 'meaning' column...")
        df['meaning'] = df['meaning'].apply(lambda x: None if x == TEMP_NULL_PLACEHOLDER else x)
        # Alternatively, using replace:
        # df['meaning'] = df['meaning'].replace(TEMP_NULL_PLACEHOLDER, None)
        print("STEP 7: COMPLETED.")
        print("DataFrame info after all filling/replacement:")
        df.info()


    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_filepath}")
        return
    except Exception as e:
        print(f"\n!!!!!!!!!! ERROR OCCURRED !!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("--------------------------------------")
        print("Traceback:")
        traceback.print_exc()
        print("--------------------------------------")
        if df is not None:
             print("DataFrame state just before error (if available):")
             try:
                 df.info()
                 print(df.head())
             except Exception as E:
                 print(f"Could not print DataFrame info/head: {E}")
        else:
             print("DataFrame ('df') was not successfully assigned.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return

    # --- Rest of the function (TypeScript generation) ---
    # No changes needed here, the logic already checks for `is not None`
    ts_content_parts = []
    if type_import:
        ts_content_parts.append(type_import)
        ts_content_parts.append("")
    ts_content_parts.append(f"export const {variable_name}: {type_annotation} = [")
    object_data_lines = []
    entries_added = 0
    for index, row in df.iterrows():
        row_id_str = str(row['id'])
        row_type_str = str(row['type'])
        row_text_str = str(row['text'])
        if not row_id_str or not row_type_str or not row_text_str:
             print(f"Warning: Skipping row {index+2} due to missing/empty id ('{row_id_str}'), type ('{row_type_str}'), or text ('{row_text_str}').")
             continue
        ts_id = escape_ts_string(row_id_str)
        ts_type = escape_ts_string(row_type_str)
        ts_text = escape_ts_string(row_text_str)
        obj_parts = [
            f"  {{ id: '{ts_id}'",
            f"type: '{ts_type}'",
            f"text: '{ts_text}'"
        ]
        # This check remains correct as we replaced placeholder with None
        if row['type'].lower() == 'scripted' and row['meaning'] is not None:
             ts_meaning = escape_ts_string(str(row['meaning']))
             obj_parts.append(f"meaning: '{ts_meaning}'")
        object_data_lines.append(", ".join(obj_parts) + " }")
        entries_added += 1
    final_ts_content = []
    if type_import:
        final_ts_content.append(ts_content_parts[0])
        final_ts_content.append(ts_content_parts[1])
    final_ts_content.append(ts_content_parts[-1])
    if entries_added > 0:
        final_ts_content.append(",\n".join(object_data_lines))
    final_ts_content.append("];")
    try:
        os.makedirs(os.path.dirname(ts_filepath), exist_ok=True)
        print(f"Saving data to TypeScript file: {ts_filepath}")
        with open(ts_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(final_ts_content) + "\n")
        print("TypeScript file saved successfully.")
    except Exception as e:
        print(f"Error writing TypeScript file: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Script Execution Started")
    print("-" * 20)

    # --- Convert TypeScript to Excel ---
    # print("Task 1: Converting TypeScript to Excel")
    # ts_to_excel(TS_FILE_PATH, EXCEL_FILE_PATH, VARIABLE_NAME)
    # print("-" * 20)

    # --- Convert Excel back to TypeScript ---
    print("Task 2: Converting Excel to TypeScript")
    if os.path.exists(EXCEL_FILE_PATH):
        excel_to_ts(EXCEL_FILE_PATH, OUTPUT_TS_FILE_PATH, VARIABLE_NAME, TYPE_IMPORT_LINE, TYPE_ANNOTATION)
    else:
        print(f"Skipping Excel to TS conversion because the input Excel file was not found or created: {EXCEL_FILE_PATH}")
    print("-" * 20)

    print("Script Execution Finished")
