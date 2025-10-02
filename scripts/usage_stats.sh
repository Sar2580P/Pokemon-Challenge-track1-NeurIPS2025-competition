# # # #!/bin/bash

# # # # Simple Smogon Stats Downloader
# # # # Uses METAMON_CACHE_DIR environment variable

# # # # Check if METAMON_CACHE_DIR is set
# # # if [ -z "$METAMON_CACHE_DIR" ]; then
# # #     echo "Error: METAMON_CACHE_DIR environment variable is not set!"
# # #     echo "Please set it with: export METAMON_CACHE_DIR='PAC-dataset'"
# # #     exit 1
# # # fi

# # # BASE_URL="https://www.smogon.com/stats/"
# # # DOWNLOAD_DIR="$METAMON_CACHE_DIR/smogon_stats"

# # # # Create download directory
# # # mkdir -p "$DOWNLOAD_DIR"
# # # cd "$DOWNLOAD_DIR"

# # # echo "Using cache directory: $METAMON_CACHE_DIR"
# # # echo "Downloading to: $DOWNLOAD_DIR"

# # # # List of folders from 2020 onwards
# # # FOLDERS=(
# # #     "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06"
# # #     "2020-07" "2020-08" "2020-09" "2020-10"
# # #     "2020-11" "2020-12" "2021-01" "2021-02" "2021-03"
# # #     "2021-04" "2021-05" "2021-06" "2021-07" "2021-08" "2021-09"
# # #     "2021-10" "2021-11" "2021-12" "2022-01" "2022-02" "2022-03"
# # #     "2022-04" "2022-05" "2022-06" "2022-07"
# # #     "2022-08" "2022-09" "2022-10" "2022-11" "2022-12"
# # #     "2023-01" "2023-02" "2023-03" "2023-04" "2023-05" "2023-06"
# # #     "2023-07" "2023-08" "2023-09" "2023-10" "2023-11" "2023-12"
# # #     "2024-01" "2024-02" "2024-03" "2024-04" "2024-05" "2024-06"
# # #     "2024-07" "2024-08" "2024-09" "2024-10" "2024-11" "2024-12"
# # #     "2025-01" "2025-02" "2025-03" "2025-04" "2025-05" "2025-06"
# # #     "2025-07"
# # # )

# # # for folder in "${FOLDERS[@]}"; do
# # #     echo "Downloading $folder..."
# # #     wget -r -np -nH -nd -P "$folder" -A "*.txt,*.json" "${BASE_URL}${folder}/"
# # # done

# # # echo "All downloads completed in: $DOWNLOAD_DIR"

# # #!/bin/bash

# # # Simple Smogon Stats Downloader
# # # Uses METAMON_CACHE_DIR environment variable

# # # Check if METAMON_CACHE_DIR is set
# # if [ -z "$METAMON_CACHE_DIR" ]; then
# #     echo "Error: METAMON_CACHE_DIR environment variable is not set!"
# #     echo "Please set it with: export METAMON_CACHE_DIR='PAC-dataset'"
# #     exit 1
# # fi

# # BASE_URL="https://www.smogon.com/stats/"
# # DOWNLOAD_DIR="$METAMON_CACHE_DIR/smogon_stats"

# # # Create download directory
# # mkdir -p "$DOWNLOAD_DIR"
# # cd "$DOWNLOAD_DIR"

# # echo "Using cache directory: $METAMON_CACHE_DIR"
# # echo "Downloading to: $DOWNLOAD_DIR"

# # # List of folders from 2020 onwards
# # FOLDERS=(
# #     "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06" 
# #     "2020-06-DLC1" "2020-07" "2020-08" "2020-09" "2020-10" 
# #     "2020-10-DLC2" "2020-11" "2020-11-H1" "2020-11-H2" "2020-12" 
# #     "2020-12-H1" "2020-12-H2" "2021-01" "2021-02" "2021-03" 
# #     "2021-04" "2021-05" "2021-06" "2021-07" "2021-08" "2021-09" 
# #     "2021-10" "2021-11" "2021-12" "2022-01" "2022-02" "2022-03" 
# #     "2022-04" "2022-05" "2022-06" "2022-07"
# # )

# # for folder in "${FOLDERS[@]}"; do
# #     echo "Downloading $folder..."
# #     wget -r -np -nH -nd -l1 -P "$folder" -A "gen1ou*,gen9ou*" "${BASE_URL}${folder}/"
# # done

# # echo "All downloads completed in: $DOWNLOAD_DIR"

# #!/bin/bash

# # Simple Smogon Stats Downloader
# # Uses METAMON_CACHE_DIR environment variable

# # Check if METAMON_CACHE_DIR is set
# if [ -z "$METAMON_CACHE_DIR" ]; then
#     echo "Error: METAMON_CACHE_DIR environment variable is not set!"
#     echo "Please set it with: export METAMON_CACHE_DIR='PAC-dataset'"
#     exit 1
# fi

# BASE_URL="https://www.smogon.com/stats/"
# DOWNLOAD_DIR="$METAMON_CACHE_DIR/smogon_stats"

# # Create download directory
# mkdir -p "$DOWNLOAD_DIR"
# cd "$DOWNLOAD_DIR"

# echo "Using cache directory: $METAMON_CACHE_DIR"
# echo "Downloading to: $DOWNLOAD_DIR"

# # List of folders from 2020 onwards
# FOLDERS=(
#     "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06" 
#     "2020-06-DLC1" "2020-07" "2020-08" "2020-09" "2020-10" 
#     "2020-10-DLC2" "2020-11" "2020-11-H1" "2020-11-H2" "2020-12" 
#     "2020-12-H1" "2020-12-H2" "2021-01" "2021-02" "2021-03" 
#     "2021-04" "2021-05" "2021-06" "2021-07" "2021-08" "2021-09" 
#     "2021-10" "2021-11" "2021-12" "2022-01" "2022-02" "2022-03" 
#     "2022-04" "2022-05" "2022-06" "2022-07"
# )

# for folder in "${FOLDERS[@]}"; do
#     echo "Downloading $folder..."
    
#     # Create subfolders for gen1ou and gen9ou
#     mkdir -p "$folder/gen1ou"
#     mkdir -p "$folder/gen9ou"
    
#     # Download gen1ou files
#     wget -r -np -nH -nd -l1 -P "temp_$folder" -A "gen1ou*" "${BASE_URL}${folder}/"
    
#     # Move and rename gen1ou files
#     for file in temp_$folder/gen1ou*; do
#         if [ -f "$file" ]; then
#             filename=$(basename "$file")
#             mv "$file" "$folder/gen1ou/${folder}_${filename}"
#         fi
#     done
    
#     # Download gen9ou files
#     wget -r -np -nH -nd -l1 -P "temp_$folder" -A "gen9ou*" "${BASE_URL}${folder}/"
    
#     # Move and rename gen9ou files
#     for file in temp_$folder/gen9ou*; do
#         if [ -f "$file" ]; then
#             filename=$(basename "$file")
#             mv "$file" "$folder/gen9ou/${folder}_${filename}"
#         fi
#     done
    
#     # Clean up temp directory
#     rm -rf "temp_$folder"
# done

# echo "All downloads completed in: $DOWNLOAD_DIR"

#!/bin/bash

# Simple Smogon Stats Downloader
# Uses METAMON_CACHE_DIR environment variable

# Check if METAMON_CACHE_DIR is set
if [ -z "$METAMON_CACHE_DIR" ]; then
    echo "Error: METAMON_CACHE_DIR environment variable is not set!"
    echo "Please set it with: export METAMON_CACHE_DIR='PAC-dataset'"
    exit 1
fi

BASE_URL="https://www.smogon.com/stats/"
DOWNLOAD_DIR="$METAMON_CACHE_DIR/smogon_stats"

# Create download directory and subfolders
mkdir -p "$DOWNLOAD_DIR/gen1ou"
mkdir -p "$DOWNLOAD_DIR/gen9ou"
cd "$DOWNLOAD_DIR"

echo "Using cache directory: $METAMON_CACHE_DIR"
echo "Downloading to: $DOWNLOAD_DIR"

# List of folders from 2020 onwards

# List of folders from 2020 onwards
FOLDERS=(
    "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06"
    "2020-07" "2020-08" "2020-09" "2020-10"
    "2020-11" "2020-12" "2021-01" "2021-02" "2021-03"
    "2021-04" "2021-05" "2021-06" "2021-07" "2021-08" "2021-09"
    "2021-10" "2021-11" "2021-12" "2022-01" "2022-02" "2022-03"
    "2022-04" "2022-05" "2022-06" "2022-07"
    "2022-08" "2022-09" "2022-10" "2022-11" "2022-12"
    "2023-01" "2023-02" "2023-03" "2023-04" "2023-05" "2023-06"
    "2023-07" "2023-08" "2023-09" "2023-10" "2023-11" "2023-12"
    "2024-01" "2024-02" "2024-03" "2024-04" "2024-05" "2024-06"
    "2024-07" "2024-08" "2024-09" "2024-10" "2024-11" "2024-12"
    "2025-01" "2025-02" "2025-03" "2025-04" "2025-05" "2025-06"
    "2025-07"
)

for folder in "${FOLDERS[@]}"; do
    echo "Downloading $folder..."
    
    # Download gen1ou files directly to gen1ou subfolder with date prefix
    wget -r -np -nH -nd -l1 -P "temp_$folder" -A "gen1ou-*" "${BASE_URL}${folder}/"
    
    # Move and rename gen1ou files
    for file in temp_$folder/gen1ou*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "gen1ou/${folder}_${filename}"
        fi
    done
    
    # Download gen9ou files directly to gen9ou subfolder with date prefix
    wget -r -np -nH -nd -l1 -P "temp_$folder" -A "gen9ou*" "${BASE_URL}${folder}/"
    
    # Move and rename gen9ou files
    for file in temp_$folder/gen9ou*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "gen9ou/${folder}_${filename}"
        fi
    done
    
    # Clean up temp directory
    rm -rf "temp_$folder"
done

echo "All downloads completed in: $DOWNLOAD_DIR"