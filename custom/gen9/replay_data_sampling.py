import os
import random
import shutil
import os
import zipfile
from pathlib import Path
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi, HfFolder

def copy_random_files_by_year(base_dir, filenames_file, count=250000):
    """
    Selects a random sample of files for specific years and copies them
    to year-specific directories.

    Args:
        base_dir (str): The base directory where all original files are stored.
        filenames_file (str): The path to the text file with all filenames.
        count (int): The number of random files to select for each year.
    """
    # Create the output directories if they don't exist
    years_to_process = [2024, 2025]
    for year in years_to_process:
        year_dir = os.path.join("data", f"gen9ou-{year}")
        os.makedirs(year_dir, exist_ok=True)
        print(f"Created directory: {year_dir}")

    # Read all filenames from the text file
    try:
        with open(filenames_file, 'r') as f:
            all_filenames = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: The file '{filenames_file}' was not found.")
        return

    # Group files by year
    yearly_files = {year: [] for year in years_to_process}
    for filename in all_filenames:
        # Assuming the year is the 4 digits after a date pattern like DD-MM-YYYY
        try:
            year_str = filename.split('_')[-2].split('-')[-1]
            year = int(year_str)
            if year in yearly_files:
                yearly_files[year].append(filename)
        except (ValueError, IndexError):
            # Skip files that don't match the expected naming convention
            continue

    # Select random files and copy them
    for year, files in yearly_files.items():
        if len(files) > count:
            print(f"Found {len(files)} files for {year}. Selecting {count} at random...")
            sample_files = random.sample(files, count)
        else:
            print(f"Found {len(files)} files for {year}. Copying all of them.")
            sample_files = files

        dest_dir = os.path.join("data", f"gen9ou-{year}")
        for filename in sample_files:
            source_path = os.path.join(base_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.copyfile(source_path, dest_path)
            except FileNotFoundError:
                print(f"Warning: Source file '{source_path}' not found. Skipping.")

    print("File copying process completed.")


def create_zip(folder_path: str):
    source_dir = Path(folder_path)
    if not source_dir.is_dir():
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return
    zip_path = source_dir.parent / f"{source_dir.name}.zip"
    print(f"Zipping content of '{source_dir}' into '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Use rglob('*') to recursively find all files in all subdirectories
        for file in source_dir.rglob('*'):
            if file.is_file():
                # Add file to the zip.
                # 'arcname' stores the file with a relative path inside the zip,
                # preventing the full directory structure from being stored.
                arcname = file.relative_to(source_dir)
                zipf.write(file, arcname)

    print(f"✅ Successfully created zip file: {zip_path}")


def create_and_upload_to_hf(
    hf_username: str,
    repo_name: str,
    local_file_paths: list[str],
    repo_type: str = "dataset"
):
    api = HfApi()
    repo_id = f"{hf_username}/{repo_name}"
    print(f"Preparing to create repository '{repo_id}'...")

    # 3. Create the repository on the Hub
    # The `exist_ok=True` flag prevents an error if the repo already exists.
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True
        )
        print(f"✅ Repository created or already exists: {repo_url}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # 4. Upload each file
    for file_path_str in local_file_paths:
        file_path = Path(file_path_str)
        
        if not file_path.is_file():
            print(f"⚠️ Warning: File not found at '{file_path_str}'. Skipping.")
            continue
        file_name = file_path.name
        print(f"\nUploading '{file_name}' to '{repo_id}'...")
        try:
            # The `upload_file` function handles the entire upload process.
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name, # The name of the file in the repo
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print(f"✅ Successfully uploaded {file_name}.")
        except Exception as e:
            print(f"❌ Error uploading {file_name}: {e}")


# def download_files_from_hf(
#     hf_username: str,
#     repo_name: str,
#     filenames: list[str],
#     destination_folder: str = "hf_downloads",
#     repo_type: str = "dataset",
#     extract_zip: bool = True,
#     delete_zip_after_extraction: bool = True
# ):
#     repo_id = f"{hf_username}/{repo_name}"
#     destination_path = Path(destination_folder)
#     destination_path.mkdir(parents=True, exist_ok=True)
#     print(f"Files will be saved to: '{destination_path.resolve()}'")
#     for filename in filenames:
#         print("-" * 30)
#         print(f"Requesting download for: '{filename}' from repo '{repo_id}'...")
        
#         try:
#             # hf_hub_download handles the download and caching.
#             # It returns the path to the downloaded file.
#             downloaded_file_path = hf_hub_download(
#                 repo_id=repo_id,
#                 filename=filename,
#                 repo_type=repo_type,
#                 local_dir=str(destination_path), # Download directly to our target folder
#                 local_dir_use_symlinks=False # Set to False to copy file to local_dir
#             )
#             print(f"✅ Successfully downloaded '{filename}'")
#             print(f"   -> Saved at: {downloaded_file_path}")

#             if extract_zip and filename.lower().endswith('.zip'):
#                 print(f"   -> Extracting '{filename}'...")
#                 try:
#                     with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
#                         zip_ref.extractall(destination_path)
#                     print(f"   ✅ Successfully extracted.")

#                     if delete_zip_after_extraction:
#                         os.remove(downloaded_file_path)
#                         print(f"   -> Deleted original zip file '{filename}'.")

#                 except zipfile.BadZipFile:
#                     print(f"   ⚠️ Could not extract '{filename}'. Not a valid zip file.")
#                 except Exception as e:
#                     print(f"   ❌ An error occurred during extraction of '{filename}': {e}")

#         except HfHubHTTPError as e:
#             print(f"❌ Could not download '{filename}'.")
#             print(f"   It may not exist in the repository, or you may not have access.")
#             print(f"   Server response: {e}")
#         except Exception as e:
#             print(f"❌ An unexpected error occurred while downloading '{filename}': {e}")


def download_files_from_hf(
    hf_username: str,
    repo_name: str,
    filenames: list[str],
    destination_folder: str = "hf_downloads",
    repo_type: str = "dataset",
    extract_zip: bool = True,
    delete_zip_after_extraction: bool = True
):
   
    repo_id = f"{hf_username}/{repo_name}"
    destination_path = Path(destination_folder)
    destination_path.mkdir(parents=True, exist_ok=True)
    print(f"Files will be saved to: '{destination_path.resolve()}'")

    # 3. Loop through the list of files and download each one
    for filename in filenames:
        print("-" * 30)
        print(f"Requesting download for: '{filename}' from repo '{repo_id}'...")
        
        try:
            # hf_hub_download handles the download and caching.
            # It returns the path to the downloaded file.
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=str(destination_path), # Download directly to our target folder
                local_dir_use_symlinks=False # Set to False to copy file to local_dir
            )
            print(f"✅ Successfully downloaded '{filename}'")
            print(f"   -> Saved at: {downloaded_file_path}")

            if extract_zip and filename.lower().endswith('.zip'):
                print(f"   -> Extracting '{filename}'...")
                try:
                    with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
                        # Check if all files in the zip are in a single root folder
                        namelist = zip_ref.namelist()
                        if not namelist:
                            print("   -> Zip file is empty.")
                            continue

                        common_prefix = os.path.commonprefix(namelist)
                        is_single_root_folder = all(name.startswith(common_prefix) for name in namelist)

                        if is_single_root_folder and common_prefix and common_prefix.endswith('/'):
                             print(f"   -> Detected single root folder '{common_prefix}'. Stripping it.")
                             for member in zip_ref.infolist():
                                 if member.is_dir():
                                     continue
                                 
                                 # Remove the root folder from the path
                                 arcname = member.filename[len(common_prefix):]
                                 if not arcname:
                                     continue # Skip the directory entry itself
                                 
                                 target_path = destination_path / arcname
                                 target_path.parent.mkdir(parents=True, exist_ok=True)
                                 
                                 # Extract the file manually
                                 with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                     target.write(source.read())
                        else:
                            # If not a single root folder, extract normally
                            print("   -> No single root folder detected. Extracting as-is.")
                            zip_ref.extractall(destination_path)

                    print(f"   ✅ Successfully extracted.")

                    if delete_zip_after_extraction:
                        os.remove(downloaded_file_path)
                        print(f"   -> Deleted original zip file '{filename}'.")

                except zipfile.BadZipFile:
                    print(f"   ⚠️ Could not extract '{filename}'. Not a valid zip file.")
                except Exception as e:
                    print(f"   ❌ An error occurred during extraction of '{filename}': {e}")

        except HfHubHTTPError as e:
            # This specific error is useful for catching "file not found" issues
            print(f"❌ Could not download '{filename}'.")
            print(f"   It may not exist in the repository, or you may not have access.")
            print(f"   Server response: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while downloading '{filename}': {e}")



from functools import partial
HF_USERNAME = "sarvagyaP"  
REPO_NAME = "pokemon-battle-data-gen9ou"
download_gen9ou_subset_parsed_replays = partial(download_files_from_hf, hf_username=HF_USERNAME , repo_name=REPO_NAME)

if __name__ == "__main__":
    # create_zip('data/gen9ou-2024')
    # create_zip('data/gen9ou-2025')
    
    # FILES_TO_UPLOAD = [
    #     "data/gen9ou-2024.zip",
    #     "data/gen9ou-2025.zip"
    # ]
    
    # create_and_upload_to_hf(
    #     hf_username=HF_USERNAME,
    #     repo_name=REPO_NAME,
    #     local_file_paths=FILES_TO_UPLOAD
    # )
    ...

    
