#!/bin/bash

# Ensure you are on the destination branch
git checkout synthesis

# Define the source branch and base destination folder
SOURCE_BRANCH="softmax_temp"
DEST_BASE="../rtl"

# List of your specific files
FILES=(
    "rtl/tr_exp/tr_exp.sv"
    "rtl/tr_exp/round.sv"
    "rtl/tr_exp/quadratic_divider.sv"
    "rtl/tr_div/tr_div.sv"
    "rtl/tr_ln/tr_ln.sv"
    "rtl/vec_mac/vec_mac_su.sv"
    "rtl/online_sum/online_sum.sv"
    "rtl/online_sum/piped_max.sv"
    "rtl/softmax/softmax_engine.sv"
)

echo "Copying files from $SOURCE_BRANCH to $DEST_BASE..."

for file in "${FILES[@]}"; do
    # Remove the leading 'rtl/' from the path so it maps correctly
    clean_path="${file#rtl/}"
    
    # Extract the directory part (e.g., 'tr_exp')
    sub_dir=$(dirname "$clean_path")
    
    # Create the target directory if it doesn't exist
    mkdir -p "$DEST_BASE/$sub_dir"
    
    # Dump the file from the source branch into the new destination
    git show "$SOURCE_BRANCH:$file" > "$DEST_BASE/$clean_path"
    
    echo "Successfully copied: $DEST_BASE/$clean_path"
done

echo "Done!"