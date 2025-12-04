#!/bin/bash

# Auto-detect NVIDIA Bus ID
BUS_ID=$(lspci -D | grep -i nvidia | grep -i -E 'vga|3d' | head -n 1 | awk '{print $1}')

echo "Targeting GPU: $BUS_ID"
echo "Attempting Driver Unbind Reset..."

# 1. Kill Display Manager (This will kill your session!)
# Detect DM
if systemctl is-active --quiet gdm3; then DM="gdm3"; fi
if systemctl is-active --quiet lightdm; then DM="lightdm"; fi
if systemctl is-active --quiet sddm; then DM="sddm"; fi

if [ ! -z "$DM" ]; then
    echo "Stopping Display Manager ($DM)..."
    systemctl stop $DM
fi

# 2. Kill stubborn processes
fuser -k -v /dev/nvidia* 2>/dev/null
killall -9 python python3 2>/dev/null

# 3. Unload Modules
rmmod nvidia_uvm
rmmod nvidia_drm
rmmod nvidia_modeset
rmmod nvidia

# 4. The Magic: Unbind instead of Remove
# This disconnects the hardware from the kernel driver cleanly
if [ -f "/sys/bus/pci/drivers/nvidia/unbind" ]; then
    echo "Unbinding GPU from driver..."
    echo "$BUS_ID" > /sys/bus/pci/drivers/nvidia/unbind
    sleep 1
fi

# 5. Load Driver (Triggers re-bind)
echo "Reloading Driver..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_drm
modprobe nvidia_modeset

# 6. Restart GUI
if [ ! -z "$DM" ]; then
    echo "Restarting $DM..."
    systemctl start $DM
fi

echo "Done. Check nvidia-smi."
