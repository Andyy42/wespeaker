#!/bin/bash

# Specify the minimum duration in seconds
min_duration=39

# Loop through all .wav files in the current directory
for file in ./**/*.wav; do
  # Get the duration of the file in seconds as a floating point number
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")

  # Use bc to compare the floating point duration with the minimum duration
  is_shorter=$(echo "$duration < $min_duration" | bc)

  # Check if the file duration is less than the minimum duration
  if [ "$is_shorter" -eq 1 ]; then
    echo "Removing $file (Duration: $duration seconds)"
    rm "$file" # Delete the file
  else
    echo "Keeping $file (Duration: $duration seconds)"
  fi
done

