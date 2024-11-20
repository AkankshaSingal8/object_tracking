import os

def rename_files_in_folder(folder_path, prefix, start_number=1):
    """
    Renames all files in the specified folder with a given prefix and sequential numbers.

    :param folder_path: Path to the folder containing the files to rename.
    :param prefix: Prefix to use for the renamed files.
    :param start_number: Starting number for the sequential naming.
    """
    try:
        files = os.listdir(folder_path)
        for count, file_name in enumerate(files, start=start_number):
            old_file_path = os.path.join(folder_path, file_name)
            
            # Skip directories
            if not os.path.isfile(old_file_path):
                continue
            
            # Extract file extension
            file_extension = os.path.splitext(file_name)[1]
            new_file_name = f"{prefix}{count}{file_extension}"
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage Example
folder_path = "quadrant1"
prefix = "file"
rename_files_in_folder(folder_path, prefix)
