#!/bin/bash
# scripts/setup_wan_emulation.sh
# Applies tc-netem rules to simulate WAN conditions inside Docker containers.
# Run this script INSIDE each worker container to throttle cross-worker traffic.
#
# Usage:
#   bash setup_wan_emulation.sh [--latency 20ms] [--bandwidth 100mbit] [--interface eth0]
#
# Defaults: 20ms latency, 100 Mbit/s bandwidth cap on eth0
#
# CS332 | Distributed SGD Project

set -euo pipefail

IFACE="${INTERFACE:-eth0}"
LATENCY="${LATENCY:-20ms}"
BANDWIDTH="${BANDWIDTH:-100mbit}"
JITTER="${JITTER:-5ms}"

echo "[tc-netem] Applying WAN emulation on $IFACE"
echo "  latency   = $LATENCY ± $JITTER"
echo "  bandwidth = $BANDWIDTH"

# Remove any existing qdisc
tc qdisc del dev "$IFACE" root 2>/dev/null || true

# Add root HTB qdisc for bandwidth shaping
tc qdisc add dev "$IFACE" root handle 1: htb default 10

# Add class with bandwidth cap
tc class add dev "$IFACE" parent 1: classid 1:10 htb \
    rate "$BANDWIDTH" ceil "$BANDWIDTH" burst 15k

# Add netem for latency + jitter under the HTB class
tc qdisc add dev "$IFACE" parent 1:10 handle 10: netem \
    delay "$LATENCY" "$JITTER" distribution normal

echo "[tc-netem] WAN emulation active. Verify with: tc qdisc show dev $IFACE"

# Show result
tc qdisc show dev "$IFACE"
