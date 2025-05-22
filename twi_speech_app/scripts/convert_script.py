import pandas as pd
import re
import os
import json
import traceback
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Section:
    id: str
    title: str
    description: str
    prompts: List[Dict]

class ScriptConverter:
    def __init__(self):
        self.type_import_line = "import { RecordingSection, ScriptPrompt } from '@/types';"
        self.variable_name = "RECORDING_SECTIONS"
        self.type_annotation = "RecordingSection[]"

    def parse_sections_from_ts(self, filepath: str) -> List[Section]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # First, extract all the prompt arrays
            script_arrays = {}

            # Find all script arrays (scriptA, scriptB, scriptC, scriptD)
            for script_name in ['scriptA', 'scriptB', 'scriptC', 'scriptD', 'spontaneousPrompts']:
                # Find the start and end of the array
                start_pattern = f"const {script_name}: ScriptPrompt[] = ["
                start_idx = content.find(start_pattern)

                if start_idx != -1:
                    # Find the closing bracket
                    start_idx += len(start_pattern)
                    bracket_count = 1
                    end_idx = start_idx

                    while bracket_count > 0 and end_idx < len(content):
                        if content[end_idx] == '[':
                            bracket_count += 1
                        elif content[end_idx] == ']':
                            bracket_count -= 1
                        end_idx += 1

                    array_content = content[start_idx:end_idx-1]

                    # Parse prompts using string splitting
                    prompts = []
                    prompt_texts = array_content.split('},')

                    for prompt_text in prompt_texts:
                        if not prompt_text.strip():
                            continue

                        # Extract prompt properties with improved regex patterns
                        id_match = re.search(r"id:\s*'([^']*)'", prompt_text)
                        type_match = re.search(r"type:\s*'([^']*)'", prompt_text)
                        # Updated text and meaning patterns to handle escape characters
                        text_match = re.search(r"text:\s*'((?:[^'\\]|\\.)*)'", prompt_text)
                        meaning_match = re.search(r"meaning:\s*'((?:[^'\\]|\\.)*)'", prompt_text)

                        if id_match and type_match and text_match:
                            prompt = {
                                'id': id_match.group(1),
                                'type': type_match.group(1),
                                'text': text_match.group(1).replace("\\'", "'")  # Handle escaped quotes
                            }
                            if meaning_match:
                                prompt['meaning'] = meaning_match.group(1).replace("\\'", "'")  # Handle escaped quotes
                            prompts.append(prompt)

                    script_arrays[script_name] = prompts
                    print(f"Extracted {len(prompts)} prompts from {script_name}")

            # Now extract the sections
            sections_data = []
            sections_start = content.find("export const RECORDING_SECTIONS: RecordingSection[] = [")
            if sections_start != -1:
                sections_content = content[sections_start:]
                section_matches = re.finditer(
                    r'\{\s*id:\s*\'([^\']*)\',\s*title:\s*\'([^\']*)\',\s*description:\s*\'([^\']*)\',\s*prompts:\s*(script[A-Z]|spontaneousPrompts)',
                    sections_content
                )

                for match in section_matches:
                    section_id = match.group(1)
                    section_title = match.group(2)
                    section_description = match.group(3)
                    prompts_var = match.group(4)

                    if prompts_var in script_arrays:
                        section = Section(
                            id=section_id,
                            title=section_title,
                            description=section_description,
                            prompts=script_arrays[prompts_var]
                        )
                        sections_data.append(section)
                        print(f"Added section {section_id} with {len(script_arrays[prompts_var])} prompts")

            return sections_data

        except Exception as e:
            print(f"Error parsing TypeScript file: {e}")
            traceback.print_exc()
            return []


    def sections_to_excel(self, sections: List[Section], excel_path: str):
        # Ensure the excel_path has .xlsx extension
        if not excel_path.endswith('.xlsx'):
            excel_path = f"{excel_path}.xlsx"

        rows = []
        for section in sections:
            print(f"Processing section {section.id} with {len(section.prompts)} prompts")
            for prompt in section.prompts:
                row = {
                    'section_id': section.id,
                    'section_title': section.title,
                    'section_description': section.description,
                    'id': prompt['id'],
                    'type': prompt['type'],
                    'text': prompt['text'],
                    'meaning': prompt.get('meaning', '')
                }
                rows.append(row)

        if not rows:
            print("Warning: No data was extracted from the TypeScript file")
            return

        df = pd.DataFrame(rows)
        columns_order = ['section_id', 'section_title', 'section_description', 'id', 'type', 'text', 'meaning']
        df = df[columns_order]

        print(f"\nValidation Summary:")
        print(f"Total sections: {len(sections)}")
        print(f"Total prompts: {len(rows)}")
        print(f"Prompts per section:")
        for section in sections:
            print(f"  {section.id}: {len(section.prompts)} prompts")

        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\nSaved {len(rows)} prompts to {excel_path}")

    def escape_ts_string(self, value: str) -> str:
        """
        Escape special characters for TypeScript string literals while preserving Unicode characters.
        Args:
            value: The string to escape
        Returns:
            Escaped string safe for TypeScript string literals
        """
        if not isinstance(value, str):
            value = str(value)

        # Only escape necessary characters
        escape_map = {
            "'": "\\'",    # Single quote
            # "\\": "\\\\",  # Backslash
        }

        # Apply escaping
        for char, escaped in escape_map.items():
            value = value.replace(char, escaped)

        return value


    def excel_to_ts_content(self, excel_path: str) -> str:
        try:
            # Read Excel file
            df = pd.read_excel(excel_path, engine='openpyxl')

            # Group by section_id to organize prompts
            sections_grouped = df.groupby('section_id')

            # Start building TypeScript content
            lines = [
                self.type_import_line,
                ""  # Empty line after import
            ]

            # First generate all script arrays
            script_arrays = {}
            for section_id, group in sections_grouped:
                if section_id == 'Spontaneous':
                    array_name = 'spontaneousPrompts'
                else:
                    array_name = f"script{section_id.replace('Script', '')}"

                prompts = []
                for _, row in group.iterrows():
                    prompt = {
                        'id': row['id'],
                        'type': row['type'],
                        'text': row['text'],
                    }
                    if pd.notna(row['meaning']):
                        prompt['meaning'] = row['meaning']
                    prompts.append(prompt)
                script_arrays[section_id] = prompts

                # Generate array declaration
                lines.append(f"const {array_name}: ScriptPrompt[] = [")

                # Add each prompt
                for prompt in prompts:
                    prompt_line = f"{{ id: '{self.escape_ts_string(prompt['id'])}'"
                    prompt_line += f", type: '{self.escape_ts_string(prompt['type'])}'"
                    prompt_line += f", text: '{self.escape_ts_string(prompt['text'])}'"
                    if 'meaning' in prompt:
                        prompt_line += f", meaning: '{self.escape_ts_string(prompt['meaning'])}'"
                    prompt_line += " },"
                    lines.append(prompt_line)

                lines.append("]")
                lines.append("")  # Empty line between arrays

            # Generate RECORDING_SECTIONS array
            lines.append("// --- Define Sections ---")
            lines.append(f"export const {self.variable_name}: {self.type_annotation} = [")

            # Add each section
            for section_id, group in sections_grouped:
                first_row = group.iloc[0]
                array_name = 'spontaneousPrompts' if section_id == 'Spontaneous' else f"script{section_id.replace('Script', '')}"

                section = f"""{{
      id: '{self.escape_ts_string(section_id)}',
      title: '{self.escape_ts_string(first_row['section_title'])}',
      description: '{self.escape_ts_string(first_row['section_description'])}',
      prompts: {array_name},
    }}"""
                lines.append(section + ",")

            lines.append("];")
            lines.append("")

            # Add utility functions
            lines.extend([
                "export const getTotalPrompts = () => {",
                "  return RECORDING_SECTIONS.reduce((total, section) => total + section.prompts.length, 0);",
                "};",
                "",
                "export const EXPECTED_TOTAL_RECORDINGS = getTotalPrompts();",
                "export const SPONTANEOUS_PROMPTS_COUNT = 8;",
                "",
                "export const getGlobalPromptIndex = (sectionIndex: number, promptInSectionIndex: number): number => {",
                "  let globalIndex = 0;",
                "  for (let i = 0; i < sectionIndex; i++) {",
                "    globalIndex += RECORDING_SECTIONS[i].prompts.length;",
                "  }",
                "  globalIndex += promptInSectionIndex;",
                "  return globalIndex;",
                "};"
            ])

            return "\n".join(lines)

        except Exception as e:
            print(f"Error converting Excel to TypeScript: {e}")
            traceback.print_exc()
            return ""

def main():
    parser = argparse.ArgumentParser(description='Convert between TypeScript and Excel formats')
    parser.add_argument('direction', choices=['to_excel', 'to_ts'], help='Conversion direction')
    parser.add_argument('--ts-file', default='twi_speech_app/constants/script.ts', help='TypeScript file path')
    parser.add_argument('--excel-file', default='recording_script.xlsx', help='Excel file path')
    parser.add_argument('--output-ts-file', help='Output TypeScript file path (for to_ts direction)')

    args = parser.parse_args()
    converter = ScriptConverter()

    try:
        if args.direction == 'to_excel':
            print(f"Reading TypeScript file: {args.ts_file}")
            sections = converter.parse_sections_from_ts(args.ts_file)
            if sections:
                converter.sections_to_excel(sections, args.excel_file)
            else:
                print("No sections were parsed from the TypeScript file")

        else:  # to_ts
            print(f"Reading Excel file: {args.excel_file}")
            output_ts_path = args.output_ts_file or args.ts_file
            ts_content = converter.excel_to_ts_content(args.excel_file)
            if ts_content:
                with open(output_ts_path, 'w', encoding='utf-8') as f:
                    f.write(ts_content)
                print(f"Successfully converted Excel to TypeScript: {output_ts_path}")
            else:
                print("Failed to convert Excel to TypeScript")

    except Exception as e:
        print(f"Error during conversion: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
