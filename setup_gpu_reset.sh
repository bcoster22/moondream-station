#!/bin/bash

# Get current user
CURRENT_USER=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/moondream-gpu-reset"
SCRIPT_PATH="/home/bcoster/.moondream-station/moondream-station/nuclear_gpu_reset.sh"

echo "Configuring passwordless sudo for GPU reset..."
echo "User: $CURRENT_USER"

# Create the sudoers entry
# We allow both nvidia-smi (legacy) and the new nuclear reset script
ENTRY="$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi, $SCRIPT_PATH"

# Use a temporary file to validate syntax before applying
TMP_FILE=$(mktemp)
echo "$ENTRY" > "$TMP_FILE"

echo "Adding rule to $SUDOERS_FILE:"
echo "$ENTRY"

# Apply with sudo (will ask for password once)
sudo cp "$TMP_FILE" "$SUDOERS_FILE"
sudo chmod 440 "$SUDOERS_FILE"
rm "$TMP_FILE"

echo "Done! You can now run 'nvidia-smi' and the reset script without a password."
echo "Testing..."
if sudo -n nvidia-smi > /dev/null 2>&1; then
    echo "Success! Passwordless access configured."
else
    echo "Error: Configuration failed."
    exit 1
fi
