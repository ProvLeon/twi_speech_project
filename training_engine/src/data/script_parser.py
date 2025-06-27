"""
Script Parser for Twi Speech Recognition Engine
Extracts prompts from TypeScript script_actual.ts file
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TwiPrompt:
    """Represents a single Twi prompt for recording"""
    id: str
    type: str
    text: str
    meaning: str
    section: str

@dataclass
class RecordingSection:
    """Represents a section of recording prompts"""
    id: str
    title: str
    description: str
    prompts: List[TwiPrompt]

class ScriptParser:
    """Parser for extracting Twi prompts from TypeScript script file"""

    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.sections: List[RecordingSection] = []
        self.all_prompts: List[TwiPrompt] = []
        self.prompts_by_id: Dict[str, TwiPrompt] = {}

    def parse_script(self) -> Tuple[List[RecordingSection], List[TwiPrompt]]:
        """Parse the TypeScript script file and extract all prompts"""
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script file not found: {self.script_path}")

        with open(self.script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract script sections
        self._extract_script_sections(content)

        # Extract recording sections metadata
        self._extract_recording_sections_metadata(content)

        # Build prompts_by_id dictionary for quick lookup
        self.prompts_by_id = {prompt.id: prompt for prompt in self.all_prompts}

        return self.sections, self.all_prompts

    def get_prompt_by_id(self, prompt_id: str) -> Optional[TwiPrompt]:
        """Get a prompt by its ID"""
        return self.prompts_by_id.get(prompt_id)

    def _extract_script_sections(self, content: str):
        """Extract script sections (scriptAU, scriptBU, etc.)"""
        # Pattern to match script arrays
        script_pattern = r'const\s+(script[A-Z]+[A-Z]*)\s*:\s*ScriptPrompt\[\]\s*=\s*\[(.*?)\];'

        matches = re.finditer(script_pattern, content, re.DOTALL)

        for match in matches:
            section_name = match.group(1)
            section_content = match.group(2)

            prompts = self._parse_prompts_from_section(section_content, section_name)
            self.all_prompts.extend(prompts)

    def _parse_prompts_from_section(self, section_content: str, section_name: str) -> List[TwiPrompt]:
        """Parse individual prompts from a section"""
        prompts = []

        # Pattern to match individual prompt objects
        prompt_pattern = r'\{\s*id:\s*[\'"]([^\'"]+)[\'"]\s*,\s*type:\s*[\'"]([^\'"]+)[\'"]\s*,\s*text:\s*[\'"]([^\'"]+)[\'"]\s*,\s*meaning:\s*[\'"]([^\'"]+)[\'"]\s*\}'

        matches = re.finditer(prompt_pattern, section_content, re.DOTALL)

        for match in matches:
            prompt_id = match.group(1)
            prompt_type = match.group(2)
            prompt_text = match.group(3)
            prompt_meaning = match.group(4)

            # Clean up escaped quotes and other characters
            prompt_text = self._clean_text(prompt_text)
            prompt_meaning = self._clean_text(prompt_meaning)

            prompt = TwiPrompt(
                id=prompt_id,
                type=prompt_type,
                text=prompt_text,
                meaning=prompt_meaning,
                section=section_name
            )
            prompts.append(prompt)

        return prompts

    def _extract_recording_sections_metadata(self, content: str):
        """Extract recording sections metadata"""
        # Pattern to match RECORDING_SECTIONS array
        sections_pattern = r'export\s+const\s+RECORDING_SECTIONS\s*:\s*RecordingSection\[\]\s*=\s*\[(.*?)\];'

        match = re.search(sections_pattern, content, re.DOTALL)
        if not match:
            print("Warning: Could not find RECORDING_SECTIONS in script file")
            return

        sections_content = match.group(1)

        # Pattern to match individual section objects
        section_pattern = r'\{\s*id:\s*[\'"]([^\'"]+)[\'"]\s*,\s*title:\s*[\'"]([^\'"]+)[\'"]\s*,\s*description:\s*[\'"]([^\'"]+)[\'"]\s*,\s*prompts:\s*([^,}]+)'

        matches = re.finditer(section_pattern, sections_content, re.DOTALL)

        for match in matches:
            section_id = match.group(1)
            section_title = match.group(2)
            section_description = match.group(3)
            prompts_variable = match.group(4).strip()

            # Find prompts for this section
            section_prompts = [p for p in self.all_prompts if p.section == prompts_variable]

            section = RecordingSection(
                id=section_id,
                title=self._clean_text(section_title),
                description=self._clean_text(section_description),
                prompts=section_prompts
            )
            self.sections.append(section)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing escape characters and normalizing"""
        # Remove escape characters
        text = text.replace('\\"', '"').replace("\\'", "'")
        text = text.replace('\\n', ' ').replace('\\t', ' ')

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def get_all_prompts(self) -> List[TwiPrompt]:
        """Get all prompts from all sections"""
        return self.all_prompts

    def get_prompts_by_section(self, section_id: str) -> List[TwiPrompt]:
        """Get prompts for a specific section"""
        for section in self.sections:
            if section.id == section_id:
                return section.prompts
        return []

    def get_total_prompts_count(self) -> int:
        """Get total number of prompts"""
        return len(self.all_prompts)

    def get_sections_summary(self) -> Dict[str, int]:
        """Get summary of prompts per section"""
        summary = {}
        for section in self.sections:
            summary[section.id] = len(section.prompts)
        return summary

    def export_to_json(self, output_path: str):
        """Export parsed prompts to JSON file"""
        data = {
            'sections': [
                {
                    'id': section.id,
                    'title': section.title,
                    'description': section.description,
                    'prompts': [
                        {
                            'id': prompt.id,
                            'type': prompt.type,
                            'text': prompt.text,
                            'meaning': prompt.meaning
                        }
                        for prompt in section.prompts
                    ]
                }
                for section in self.sections
            ],
            'total_prompts': len(self.all_prompts),
            'sections_summary': self.get_sections_summary()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print a summary of parsed prompts"""
        print(f"📊 Script Parsing Summary")
        print(f"{'='*50}")
        print(f"Total Prompts: {len(self.all_prompts)}")
        print(f"Total Sections: {len(self.sections)}")
        print("\nSections:")
        for section in self.sections:
            print(f"  • {section.title}: {len(section.prompts)} prompts")
        print(f"{'='*50}")

def main():
    """Test the script parser"""
    import sys
    import os

    # Get script path from command line or use default
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
    else:
        # Default path assuming we're in simple_speech_engine directory
        script_path = "../training_engine/script_actual.ts"

    if not os.path.exists(script_path):
        print(f"Error: Script file not found at {script_path}")
        print("Please provide the correct path to script_actual.ts")
        sys.exit(1)

    try:
        parser = ScriptParser(script_path)
        sections, prompts = parser.parse_script()

        parser.print_summary()

        # Export to JSON
        parser.export_to_json("parsed_prompts.json")
        print("\n✅ Prompts exported to parsed_prompts.json")

        # Show first few prompts as example
        print("\n📝 Sample Prompts:")
        for i, prompt in enumerate(prompts[:3]):
            print(f"  {i+1}. [{prompt.id}] {prompt.text}")
            print(f"     Meaning: {prompt.meaning}")
            print()

    except Exception as e:
        print(f"Error parsing script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
