#!/bin/bash

# This script runs all tests sequentially

# Initialize a variable to track the overall test result
all_tests_passed=true

# Function to build the find command
build_find_command() {
    local find_command="find . -type f -name \"test_*.py\""
    echo "$find_command"
}

# Build the find command
find_command=$(build_find_command)

# Evaluate the find command and run pytest on each file
eval $find_command | while read -r file; do
    echo "Running pytest on $file"
    pytest "$file"

    # Check the exit code of pytest
    if [ $? -ne 0 ]; then
        all_tests_passed=false
    fi
done

# Final result
if [ "$all_tests_passed" = true ]; then
    echo "All tests passed!"
else
    echo "Some tests failed."
    exit 1
fi