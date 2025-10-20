#!/bin/bash

# Check for jq
command -v jq >/dev/null 2>&1 || { echo >&2 "This script requires 'jq'. Please install it first."; exit 1; }

echo -e "\nAnalyzing JSON files in r-*/ directories...\n"

# Create temp files
combined_json="combined_temp.jsonl"
log_file="process_log.txt"
> "$combined_json"
> "$log_file"

# Enable strict globbing
shopt -s nullglob

# Counters
total_rows=0
file_count=0
max_rows=0
min_rows=-1
max_file=""
min_file=""

# Collect r-* directories
dirs=(r-*/)

# Process each directory
for dir in "${dirs[@]}"; do
    [ -d "$dir" ] || continue
    subreddit=$(basename "$dir")
    json_file=$(find "$dir" -maxdepth 1 -type f -name "*.json" | sort | head -n 1)
    [ -f "$json_file" ] || {
        echo "Skipping $subreddit: No JSON file found." | tee -a "$log_file"
        continue
    }

    echo "==== START: $subreddit ====" | tee -a "$log_file"
    echo "Processing: $json_file"

    # Count rows (rough)
    row_count=$(awk '$0 ~ /[^[:space:]]/' "$json_file" | wc -l)
    total_rows=$((total_rows + row_count))
    file_count=$((file_count + 1))
    [ "$row_count" -gt "$max_rows" ] && { max_rows=$row_count; max_file=$json_file; }
    [ "$min_rows" -eq -1 ] || [ "$row_count" -lt "$min_rows" ] && { min_rows=$row_count; min_file=$json_file; }

    # Process content
    if jq -e 'type == "array" and length > 0' "$json_file" >/dev/null 2>&1; then
        first_type=$(jq -r '.[0] | type' "$json_file")
        if [ "$first_type" = "string" ]; then
            jq -c --arg sr "$subreddit" --arg sr_full "r/$subreddit" '
                .[] | {body: ., subreddit: $sr, subreddit_full: $sr_full}
            ' "$json_file" >> "$combined_json" || echo "  ⚠ Error processing $json_file as string array" | tee -a "$log_file"
        elif [ "$first_type" = "object" ]; then
            jq -c --arg sr "$subreddit" --arg sr_full "r/$subreddit" '
                .[] | .subreddit = $sr | .subreddit_full = $sr_full
            ' "$json_file" >> "$combined_json" || echo "  ⚠ Error processing $json_file as object array" | tee -a "$log_file"
        else
            echo "Skipping $json_file: Unhandled array element type: $first_type" | tee -a "$log_file"
            continue
        fi
    else
        echo "Skipping $json_file: Not a valid JSON array" | tee -a "$log_file"
        continue
    fi

    echo "==== END: $subreddit ====" | tee -a "$log_file"
done

# Summary
if [ "$file_count" -eq 0 ]; then
    echo -e "\nNo valid JSON files found.\n"
    rm -f "$combined_json"
    exit 1
fi

avg=$(awk "BEGIN {printf \"%.2f\", $total_rows / $file_count}")
echo -e "\nTotal JSON files: $file_count"
echo "Total Data Lines: $total_rows"
echo "Average lines per JSON: $avg"
echo "Max lines: $max_rows in $max_file"
echo "Min lines: $min_rows in $min_file"

# Ask user to combine and export
read -rp $'\nDo you want to combine and save all data in full-data/ [y/n]? ' answer
if [ "$answer" != "y" ]; then
    echo -e "\nExiting. Combined data not saved."
    rm -f "$combined_json"
    exit 0
fi

echo -e "\nCreating 'full-data/' with JSON and CSV output..."
mkdir -p full-data

# Save full JSON array
jq -s '.' "$combined_json" > full-data/combined.json

# Extract headers dynamically
csv_fields=$(jq -r 'reduce inputs as $item ({}; . * $item) | keys_unsorted | @csv' "$combined_json" | head -n1 | tr -d '"')

# Create CSV
jq -r --argjson header "[$csv_fields]" '
  $header as $h |
  ($h | @csv),
  (.[]
    | [.[$h[]]?] | @csv)
' "$combined_json" > full-data/combined.csv

# Final stats
final_lines=$(wc -l < "$combined_json")
echo -e "\nFull Combined Data Summary:"
echo "----------------------------"
echo "Combined Data Rows: $final_lines"
echo "Original Total Rows: $total_rows"
echo "Difference: $((final_lines - total_rows))"
echo ""
echo "Saved:"
echo " - full-data/combined.json"
echo " - full-data/combined.csv"
echo " - Logs in: $log_file"

# Clean up
rm -f "$combined_json"
