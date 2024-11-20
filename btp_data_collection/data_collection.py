import subprocess

# Define the path to the Hector_Initialisation.py file
script_path = "Hector_Initialisation.py"  # Ensure the script is in the same directory or provide the full path

# Define input values
position_x = [1.0, 2.0, 3.0]
position_y = [1.0, 2.0, 3.0]
position_z = [3.0, 3.0, 3.0]
orientation_x = [0.0, 0.0, 0.0]
orientation_y = [0.0, 0.0, 0.0]
orientation_z = [1.0, 1.0, 1.0]
orientation_w = [0.0, 0.0, 0.0]

# Run the script with arguments
try:
    subprocess.run(
        ["python", script_path, str(position_x), str(position_y), str(position_z), str(orientation_x), str(orientation_y), str(orientation_z), str(orientation_w)],
        check=True
    )
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
