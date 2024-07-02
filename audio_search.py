import os
import subprocess

def run_mfa(audio_dir, text_dir, language_model, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the MFA command
    mfa_command = [
        'mfa', 'align', 
        audio_dir, 
        text_dir, 
        language_model, 
        output_dir
    ]

    # Run the command
    process = subprocess.run(mfa_command, capture_output=True, text=True)

    # Check for errors
    if process.returncode != 0:
        print(f"Error running MFA: {process.stderr}")
    else:
        print(f"Alignment completed successfully. Output saved to {output_dir}")

# Example usage
audio_directory = '/path/to/audio'
text_directory = '/path/to/text'
language = 'english'  # Replace with your desired language model
output_directory = '/path/to/output'

run_mfa(audio_directory, text_directory, language, output_directory)
