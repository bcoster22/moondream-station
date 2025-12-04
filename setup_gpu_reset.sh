#!/bin/bash

# Get current user
CURRENT_USER=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/moondream-gpu-reset"

echo "Configuring passwordless sudo for nvidia-smi..."
echo "User: $CURRENT_USER"

# Create the sudoers entry
ENTRY="$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi"

# Use a temporary file to validate syntax before applying
TMP_FILE=$(mktemp)
echo "$ENTRY" > "$TMP_FILE"

echo "Adding rule to $SUDOERS_FILE:"
echo "$ENTRY"

# Apply with sudo (will ask for password once)
sudo cp "$TMP_FILE" "$SUDOERS_FILE"
sudo chmod 440 "$SUDOERS_FILE"
rm "$TMP_FILE"

echo "Done! You can now run 'nvidia-smi' without a password."
echo "Testing..."
if sudo -n nvidia-smi > /dev/null 2>&1; then
    echo "Success! Passwordless access configured."
else
    echo "Error: Configuration failed."
    exit 1
fi
