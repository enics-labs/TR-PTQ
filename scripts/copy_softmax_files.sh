#!/bin/bash

# 1. Check if the user provided the input list file
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_dut_src_list.txt>"
    echo "Example: $0 ../inputs/dut_src_list_vec_mac_su.txt"
    exit 1
fi

LIST_FILE=$1

if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: Input file '$LIST_FILE' not found!"
    exit 1
fi



# 2. Define the source branch and base destination folder
SOURCE_BRANCH="tr_core"
DEST_BASE="../rtl_cpy"

# Ensure you are on the destination branch
git checkout softmax_synthesis

# 3. Clear the synthesis RTL folder
rm -rf "$DEST_BASE"
mkdir -p "$DEST_BASE"

echo "Copying files from $SOURCE_BRANCH to $DEST_BASE..."

# 4. Read the list file line by line
while IFS= read -r FILE || [ -n "$FILE" ]; do
    # Strip carriage returns and whitespace
    FILE=$(echo "$FILE" | tr -d '\r' | xargs)
    
    # Skip empty lines and comments
    if [[ -z "$FILE" || "$FILE" == \#* || "$FILE" == //* ]]; then
        continue
    fi

    # Strip any leading '../' from the path
    while [[ "$FILE" == ../* ]]; do
        FILE="${FILE#../}"
    done 
    
    # Remove the leading 'rtl/' from the path so it maps correctly into DEST_BASE
    clean_path="${FILE#rtl/}"
    
    # Extract the directory part (e.g., 'tr_exp')
    sub_dir=$(dirname "$clean_path")
    
    # Create the target directory if it doesn't exist
    mkdir -p "$DEST_BASE/$sub_dir"
    
    # Dump the file from the source branch into the new destination
    # We route stderr to /dev/null to catch missing files cleanly
    if git show "$SOURCE_BRANCH:$FILE" > "$DEST_BASE/$clean_path" 2>/dev/null; then
        echo "Successfully copied: $DEST_BASE/$clean_path"
    else
        echo "WARNING: File '$FILE' not found in branch '$SOURCE_BRANCH'! Skipping..."
        rm -f "$DEST_BASE/$clean_path" # Clean up the empty file created by the redirect
    fi

done < "$LIST_FILE"

echo "------------------------------------------------"
echo "Done! Target '$DEST_BASE' has been freshly populated."
echo "------------------------------------------------"