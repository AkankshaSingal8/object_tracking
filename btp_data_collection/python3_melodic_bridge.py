import subprocess
import json

def run_subprocess(model_script):
    try:
        # The command to run a Python script using python3
        command = ["python3", model_script]

        # Run the subprocess and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Parse the JSON output from the subprocess
        output_list = json.loads(result.stdout)

        # Print the captured list
        print("Captured list from subprocess:", output_list)

        return output_list  # Return the list to the caller
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)
        print("Subprocess stderr:", e.stderr)
        return None
    except json.JSONDecodeError as e:
        print("Failed to parse JSON output:", e)
        return None

if __name__ == "__main__":
    # Specify the subprocess script name
    model_script = "model_trial.py"

    # Run the subprocess and capture its return value
    captured_values = run_subprocess(model_script)

    # If successful, print the final list
    if captured_values is not None:
        print("Final captured values:", captured_values)
