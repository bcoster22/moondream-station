#!/bin/bash

# --- 1. PREPARATION & AUTO-DETECT ---
echo "--- NUCLEAR GPU RESET SCRIPT ---"

# Detect the ID. We look for NVIDIA. 
# We use 'lspci -n' to get raw IDs, which helps if the device is glitching.
GPU_BUS_ID=$(lspci -D | grep -i nvidia | grep -i -E 'vga|3d' | head -n 1 | awk '{print $1}')

# Check if we actually found a GPU
if [ -z "$GPU_BUS_ID" ]; then
    echo "⚠️  STATUS: GPU not detected in lspci."
    echo "   The GPU may have already fallen off the bus."
    echo "   Proceeding directly to Global Rescan to try and recover it."
    DEVICE_PRESENT=false
else
    echo "✅ STATUS: GPU detected at Bus ID: $GPU_BUS_ID"
    DEVICE_PRESENT=true
fi

echo "!!! WARNING: This simulates physically unplugging the GPU."
echo "!!! This allows you to recover from Xid 79 / 'GPU has fallen off the bus' errors."
# REMOVED INTERACTIVE PROMPT FOR API USAGE
# read -p "Press Enter to NUKE the GPU connection..."

# --- 2. CLEANUP SOFTWARE STACK ---
echo "1. Killing processes..."
fuser -k -v /dev/nvidia* >/dev/null 2>&1
killall -9 python python3 >/dev/null 2>&1

# Stop GUI (Optimized for your Linux system)
if systemctl is-active --quiet gdm3; then systemctl stop gdm3; DM="gdm3"; fi
if systemctl is-active --quiet lightdm; then systemctl stop lightdm; DM="lightdm"; fi
if systemctl is-active --quiet sddm; then systemctl stop sddm; DM="sddm"; fi

echo "2. Unloading drivers..."
# We ignore errors here because if the GPU is crashed, modules might be stuck.
rmmod -f nvidia_uvm >/dev/null 2>&1
rmmod -f nvidia_drm >/dev/null 2>&1
rmmod -f nvidia_modeset >/dev/null 2>&1
rmmod -f nvidia >/dev/null 2>&1

# --- 3. THE NUCLEAR RESET (HARDWARE LAYER) ---

if [ "$DEVICE_PRESENT" = true ]; then
    echo "3. Removing GPU device node ($GPU_BUS_ID)..."
    # This tells the kernel to stop talking to the hardware immediately
    echo 1 > /sys/bus/pci/devices/$GPU_BUS_ID/remove
    sleep 2
fi

echo "4. Performing Global PCIe Rescan..."
# This tells the OS to scan all PCIe lanes for hardware changes
echo 1 > /sys/bus/pci/rescan
echo "   Waiting for bus to settle (5 seconds)..."
sleep 5

# --- 4. VERIFICATION & RESTORE ---

# Check if it came back
NEW_BUS_ID=$(lspci -D | grep -i nvidia | grep -i -E 'vga|3d' | head -n 1 | awk '{print $1}')

if [ -z "$NEW_BUS_ID" ]; then
    echo "❌ CRITICAL: GPU did not reappear after rescan."
    echo "   The hardware may be completely frozen. A physical reboot is required."
    exit 1
else
    echo "✅ SUCCESS: GPU found at $NEW_BUS_ID"
fi

echo "5. Reloading Drivers..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_drm
modprobe nvidia_modeset

# Run a quick check
echo "6. Health Check (nvidia-smi)..."
nvidia-smi

# Restart GUI if we stopped it
if [ ! -z "$DM" ]; then
    echo "7. Restarting GUI ($DM)..."
    systemctl start $DM
fi
