#!/bin/bash

# Duration of each split in seconds (not needed directly in the ffmpeg command but kept for reference)
split_duration=40
# Output file prefix (will be used in constructing the ffmpeg command)
output_prefix="SPLIT_"
# new dir
# new_dir="/scratch/project_465000792/xodehn09/data/NAKI_SPLIT"
new_dir="/pfs/lustrep1/scratch/project_465000792/xodehn09/data/NAKI_filtered/test/NAKI_split"

# Function to split a single audio file using ffmpeg
split_audio_file() {
    local audio_file="$1"
    # Extract directory and base filename for output naming
    local dir=$(dirname "$audio_file")
    local base=$(basename "$audio_file" .wav)

    # Constructing the output pattern including directory and base filename
    local output_pattern="${new_dir}/${dir}/${base}_${output_prefix}%03d.wav"
    mkdir -p "${new_dir}/${dir}"

    # Execute ffmpeg command to split the audio file
    ffmpeg -i "$audio_file" -f segment -segment_time $split_duration -c copy "$output_pattern"
}

export -f split_audio_file
export new_dir
export split_duration
export output_prefix

# Find all .wav files recursively from the current directory and process them in parallel
find . -type f -name "*.wav" | parallel -j 32 split_audio_file


