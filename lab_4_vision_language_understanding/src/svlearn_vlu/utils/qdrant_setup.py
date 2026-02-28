#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import subprocess
from svlearn_vlu import config
CONTAINER_NAME = config['vector-db']['container']
PORT = config['vector-db']['port']
#  -------------------------------------------------------------------------------------------------

def is_qdrant_running():
    """Check if the Qdrant container is already running."""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    return CONTAINER_NAME in result.stdout.strip()

#  -------------------------------------------------------------------------------------------------
def start_vector_db():
    """Start Qdrant container if not already running."""

    qdrant_mount_dir = config["vector-db"]["volume"]
    print(f"Qdrant mount directory: {qdrant_mount_dir}")

    if is_qdrant_running():
        print(f"Qdrant container with name: {CONTAINER_NAME} is already running.")
        return

    print("Starting Qdrant container...")
    subprocess.run([
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "-p", f"{PORT}:6333", "-p", "6334:6334",
        "-v", f"{qdrant_mount_dir}:/qdrant/storage",
        "qdrant/qdrant:v1.10.0"
    ])

#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_vector_db()