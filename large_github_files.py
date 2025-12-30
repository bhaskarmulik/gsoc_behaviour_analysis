import os

def find_large_files(directory, size_limit_mb=100):
    size_limit_bytes = size_limit_mb * 1024 * 1024
    large_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > size_limit_bytes:
                large_files.append(file_path)

    return large_files

def add_to_gitignore(files, gitignore_path=".gitignore"):
    with open(gitignore_path, "a") as gitignore_file:
        for file in files:
            gitignore_file.write(f"{file}\n")

if __name__ == "__main__":
    directory_to_search = "./"
    large_files = find_large_files(directory_to_search)

    if large_files:
        print("Files exceeding the GitHub file size limit:")
        for file in large_files:
            print(file)
        add_to_gitignore(large_files)
        print("The above files have been added to .gitignore.")
    else:
        print("No files exceed the GitHub file size limit.")