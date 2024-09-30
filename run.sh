#!/bin/bash

# Needed for MacOS to work run the `pkill` properly
export LC_CTYPE=C 
export LANG=C

# Kill any currently running client.py processes
pkill -f 'flower-client-app'

# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink'

# Start the flower server
echo "[RUNNER] Starting flower server in background..."
flower-superlink --insecure > /dev/null 2>&1 &
sleep 2

# Parition Types   
 export PARTITION_TYPE="noniid"  # (1)-(2)-(3);
# export PARTITION_TYPE="iid" # (123)-(123)-(123);
# export PARTITION_TYPE="vertical" # per-feature (only 2 clients)

echo "[RUNNER] Using $PARTITION_TYPE Partition Type"

# Aggregation Type
# export AGGREGATION_TYPE="regular"
 export AGGREGATION_TYPE="secure"

echo "[RUNNER] Using $AGGREGATION_TYPE Aggregation Type"

# Number of client processes to start
export N_CLIENTS=3 

if [[ $PARTITION_TYPE == "vertical" ]]; then
    N_CLIENTS=2 && echo "[RUNNER] Starting 2 ClientApps"
else
    echo "[RUNNER] Starting $N_CLIENTS ClientApps"
fi

# Start resource measurement in the background
source ./.measure_resource.sh
measure_resources &
bg_pid=$!

echo "[RUNNER] Background Process PID: $bg_pid" 

# Start N client processes
for i in $(seq 1 $N_CLIENTS)
do
  export CID=$((i-1)) &&
  flower-client-app --insecure client:app &
  sleep 0.1
done

echo "[RUNNER] Starting ServerApp..."
flower-server-app --insecure server:app --verbose




echo "[RUNNER] Clearing background processes..."

# Kill any currently running client.py processes
pkill -f 'flower-client-app'

# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink'


pkill -P $bg_pid
pkill -P $$
