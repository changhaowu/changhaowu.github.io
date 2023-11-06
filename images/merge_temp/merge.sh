#!/bin/bash

# The directory containing your images
image_directory="images/merge_temp"

# Ensure that you actually have .png files in the directory
echo "Looking for PNG files in $image_directory"

file_count=$(ls -1 "$image_directory"/*.png 2>/dev/null | wc -l)
if [ "$file_count" -eq 0 ]; then
    echo "No PNG files found in the directory."
    exit 1
fi

# Find the maximum height among all PNG images
max_height=0
for image in "$image_directory"/*.png; do
    echo "Processing $image"
    height=$(identify -format "%h" "$image")
    if [ -z "$height" ]; then
        echo "Unable to get height for $image"
        continue
    fi
    echo "Height of $image is $height"
    if (( height > max_height )); then
        max_height=$height
    fi
done

echo "Maximum height is $max_height pixels"

# Resize images proportionally to the maximum height and concatenate them horizontally
for image in "$image_directory"/*.png; do
    basename=$(basename "$image")
    echo "Resizing $basename"

    # Skip resizing if this is the output file from a previous run
    if [[ "$basename" == "combined.png" ]]; then
        echo "Skipping $basename"
        continue
    fi

    # Get image width and height
    width=$(identify -format "%w" "$image")
    height=$(identify -format "%h" "$image")

    # Calculate new width to maintain aspect ratio
    new_width=$(( width * max_height / height ))

    # Resize the image to the new width and maximum height
    # Use "!" to ignore aspect ratio if you want to force it to the exact max height size
    resized_image_path="$image_directory/resized_${basename}"
    convert "$image" -resize "${new_width}x$max_height!" "$resized_image_path"

    # Log the success of the operation
    if [ $? -eq 0 ]; then
        echo "Resized image saved to $resized_image_path"
    else
        echo "Failed to resize $image"
    fi
done

# Concatenate the resized images
output_image="$image_directory/combined.png"
convert "$image_directory"/resized_*.png +append "$output_image"

# Check if the output image was created successfully
if [ -f "$output_image" ]; then
    echo "Images have been combined into $output_image"
    # Optionally, remove the resized images
    # rm "$image_directory"/resized_*.png
else
    echo "Failed to combine images"
fi
