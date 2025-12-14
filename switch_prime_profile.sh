#!/bin/bash
# Usage: ./switch_prime_profile.sh [nvidia|on-demand|intel]

if [ -z "$1" ]; then
    echo "Usage: $0 [nvidia|on-demand|intel]"
    exit 1
fi

# Validate input to prevent command injection
if [[ "$1" != "nvidia" && "$1" != "on-demand" && "$1" != "intel" ]]; then
    echo "Invalid profile. Allowed: nvidia, on-demand, intel"
    exit 1
fi

prime-select "$1"
