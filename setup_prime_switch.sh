#!/bin/bash

# Determine the actual user (even if run with sudo)
if [ -n "$SUDO_USER" ]; then
    CURRENT_USER="$SUDO_USER"
else
    CURRENT_USER=$(whoami)
fi

# Prevent running as root if not via sudo (unlikely, but safe)
if [ "$CURRENT_USER" == "root" ]; then
    echo "Warning: Configuring for 'root' user. If this is unintended, run this script as your normal user (with sudo if needed)."
fi

TARGET_DIR="/home/bcoster/.moondream-station/moondream-station"
TARGET_SCRIPT="$TARGET_DIR/switch_prime_profile.sh"
SUDOERS_FILE="/etc/sudoers.d/moondream-prime-switch"

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: Could not find switch_prime_profile.sh at $TARGET_SCRIPT"
    exit 1
fi

echo "Configuring passwordless sudo for Prime Switch script..."
echo "User: $CURRENT_USER"
echo "Script: $TARGET_SCRIPT"

# Create the sudoers entry
ENTRY="$CURRENT_USER ALL=(ALL) NOPASSWD: $TARGET_SCRIPT"

# Use a temporary file to validate syntax
TMP_FILE=$(mktemp)
echo "$ENTRY" > "$TMP_FILE"

echo "Adding rule to $SUDOERS_FILE:"
echo "$ENTRY"

# Apply with sudo (we might already be root, but sudo handles elevation if not)
if [ "$EUID" -ne 0 ]; then
    sudo cp "$TMP_FILE" "$SUDOERS_FILE"
    sudo chmod 440 "$SUDOERS_FILE"
else
    cp "$TMP_FILE" "$SUDOERS_FILE"
    chmod 440 "$SUDOERS_FILE"
fi
rm "$TMP_FILE"

# Make the target script executable
chmod +x "$TARGET_SCRIPT"

echo "Done! You can now run the Prime Switch from the UI."
