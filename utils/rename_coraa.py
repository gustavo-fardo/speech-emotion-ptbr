import os

folder_path = "data_train/train"
files = os.listdir(folder_path)

# Iterate through each file and rename
for i, file_name in enumerate(files):
    # print(file_name[-11:-4])
    print(file_name)
    if file_name[-11:-4] == "neutral":
        # Construct the new file name
        new_name = f"CORn{i + 1}{os.path.splitext(file_name)[1]}"
        # Build the full path for the old and new file names
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        print(new_name)

        # Rename the file
        os.rename(old_path, new_path)