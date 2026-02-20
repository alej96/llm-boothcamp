#!/bin/bash

# Get the current working directory
curr_working_dir=$(pwd)

# Check if .env.example exists
if [ ! -f ".env.example" ]; then
    echo "Error: .env.example file not found in current directory"
    exit 1
fi

# Copy .env.example to .env
cp .env.example .env

# Replace the three specific variables in .env
sed -i '' "s|^export BOOTCAMP_ROOT_DIR=.*|export BOOTCAMP_ROOT_DIR=\"${curr_working_dir}\"|" .env
sed -i '' "s|^export PROJECT_PYTHON=.*|export PROJECT_PYTHON=\"${curr_working_dir}/.venv/bin/python\"|" .env
sed -i '' "s|^export PYTHONPATH=.*|export PYTHONPATH=\"${curr_working_dir}/src\"|" .env
