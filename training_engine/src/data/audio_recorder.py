#!/usr/bin/env python3
"""
Audio Recording Interface for the Superior Twi Speech Training Engine
This script uses the TwiDataManager to handle audio recording sessions.
"""


import os
import sys
from pathlib import Path

# Ensure the source directory is in the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_dir = src_dir.parent

# Add both paths to handle different import scenarios
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    # Try importing with src prefix first (when imported from main_engine)
    from src.data.data_manager import TwiDataManager
    from src.config.config_loader import load_config
except ImportError:
    try:
        # Fallback to relative imports (when run directly)
        from data.data_manager import TwiDataManager
        from config.config_loader import load_config
    except ImportError:
        try:
            # Alternative: Try absolute imports
            from twi_speech.training_engine.src.data.data_manager import TwiDataManager
            from twi_speech.training_engine.src.config.config_loader import load_config
        except ImportError as e:
            print(f"Error importing modules: {e}")
            print("Please ensure you are running this script from the 'training_engine' directory or that the project structure is correct.")
            sys.exit(1)

def interactive_recording_session(config=None):
    """
    Guides the user through an interactive audio recording session.

    Args:
        config: Optional configuration dictionary. If not provided, will load from default location.
    """
    print("🚀 Welcome to the Twi Speech Audio Recorder")
    print("==============================================")

    # Load configuration if not provided
    if config is None:
        # Assuming a default config file path, adjust if necessary
        config_path = current_dir.parent.parent / "configs" / "default_config.yaml"
        if not config_path.exists():
            print(f"Warning: Default config file not found at {config_path}")
            # Create a dummy config if none exists, as TwiDataManager might need it
            config = {
                'data_split': {},
                'mongodb': {},
                'r2_storage': {},
                'audio': {},
                'text': {},
                'cache_dir': 'data/cache',
                'recordings_dir': 'data/recordings',
                'script_file': str(src_dir.parent / 'script_actual.ts')
            }
        else:
            config = load_config(config_path)


    # Initialize the data manager
    try:
        data_manager = TwiDataManager(config)
        if not data_manager.script_parser:
            print("\n❌ Critical Error: Could not load recording prompts from script file.")
            print(f"   Please ensure '{data_manager.script_path}' exists and is correctly formatted.")
            return
    except Exception as e:
        print(f"\n❌ Failed to initialize Data Manager: {e}")
        return

    # Get speaker ID
    speaker_id = input("\n👤 Please enter your Speaker ID (e.g., 'john_doe'): ").strip()
    if not speaker_id:
        print("❌ Speaker ID cannot be empty. Exiting.")
        return

    # Start the session
    data_manager.start_session(speaker_id)

    try:
        while True:
            print("\n--------------------")
            print("🎤 Recording Menu")
            print("--------------------")
            print("1. Record a specific section")
            print("2. Record all remaining sections")
            print("3. Finish and save session")
            print("--------------------")

            choice = input("Enter your choice (1-3): ").strip()

            if choice == '1':
                print("\nAvailable Sections:")
                for i, section in enumerate(data_manager.sections, 1):
                    print(f"  {i}. {section.title} ({len(section.prompts)} prompts)")

                try:
                    section_num_str = input("\nSelect the section number you want to record: ")
                    section_num = int(section_num_str) - 1
                    if 0 <= section_num < len(data_manager.sections):
                        selected_section = data_manager.sections[section_num]
                        data_manager.record_section(selected_section.id, speaker_id)
                    else:
                        print("❌ Invalid section number.")
                except ValueError:
                    print("❌ Please enter a valid number.")
                except (EOFError, KeyboardInterrupt):
                    print("\nRecording interrupted.")
                    break

            elif choice == '2':
                print("\nRecording all available sections...")
                for section in data_manager.sections:
                    # A more robust check would be to see which prompts have been recorded
                    # For simplicity, we just iterate through all sections here
                    print(f"\n--- Starting Section: {section.title} ---")
                    data_manager.record_section(section.id, speaker_id)
                    print(f"--- Finished Section: {section.title} ---")
                print("\n✅ All sections have been presented for recording.")
                break # Exit after trying all sections

            elif choice == '3':
                print("\nFinishing session...")
                break

            else:
                print("❌ Invalid choice. Please select a number from 1 to 3.")

    except (EOFError, KeyboardInterrupt):
        print("\n\nSession interrupted by user.")
    finally:
        # Finish and save the session data
        data_manager.finish_session()
        print("\n🎉 Recording session has been successfully saved!")
        print("Thank you for your contribution.")

if __name__ == "__main__":
    interactive_recording_session()
