import os
import shutil
import sys

if len(sys.argv) != 3:
    print('Invalid argument count! Please pass source directory and destination directory after the file name')
    sys.exit()

# Get the source and destination paths from arguments
source = os.path.abspath(sys.argv[1])  # Convert to absolute path
dest = os.path.abspath(sys.argv[2])    # Convert to absolute path

print(f"Source directory: {source}")
print(f"Destination directory: {dest}")

# Check if source and destination exist
if not os.path.exists(source):
    print(f"Source directory does not exist: {source}")
    sys.exit()
if not os.path.exists(dest):
    print(f"Destination directory does not exist: {dest}")
    sys.exit()

# List of files to move
try:
    with open('files_to_sync.txt', 'r') as f:
        files_to_move = f.read().splitlines()
except FileNotFoundError:
    print("files_to_sync.txt not found!")
    sys.exit()

# Copy files from source to destination
try:
    for file in files_to_move:
        src_path = os.path.join(source, file)
        dest_path = os.path.join(dest, file)

        # Ensure destination directories exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        print(f"Copying file: {src_path} -> {dest_path}")
        shutil.copy(src_path, dest_path)

    print(f"Finished copying {len(files_to_move)} files.")
except Exception as e:
    print(f"Error occurred while copying files: {e}")