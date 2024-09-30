#!/bin/bash

unset -f measure_resources

# CSV file to write the resource usage
output_file="./saves/resource_usage.csv"

# Write CSV header
echo -e "Time\tPID\t%CPU\tMEM(MB)\tCommand" > $output_file



# Function to measure CPU and RAM usage
measure_resources() {
    while true; do
        # Get current timestamp
        timestamp=$(date +"%Y-%m-%dT%H:%M:%S")
        
        
        # Get process information containing the keyword 'flwr' and append to CSV
        ps -eo ppid,%cpu,rss,command | grep 'flower' | grep -v grep | awk -v timestamp="$timestamp" '{
            print timestamp "\t" $1 "\t" $2 "\t" $3/1024 "\t" $5
        }' >> $output_file
        
        sleep 1
    done
}

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    
    
    unset -f measure_resources
    # Modify CSV header and function to include GPU-related measurements
    echo -e "Time\tPID\t%CPU\tMEM(MB)\tGPU_Load(%)\tVRAM_Usage(MB)\tCommand" > $output_file
    
    measure_resources() {
        while true; do
            # Get current timestamp
            timestamp=$(date +"%Y-%m-%dT%H:%M:%S")
            	    
   	    # Get GPU load and VRAM usage
    	    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
            # Get GPU load and VRAM usage
            gpu_load=$(echo "$gpu_info" | cut -d',' -f1)
            vram_usage=$(echo "$gpu_info" | cut -d',' -f2)
            # ppid gives absolete cpu usage for all cores. thats why values are above 100 
            # Get process information containing the keyword 'flwr' and append to CSV
            ps -eo ppid,%cpu,rss,command | grep 'flower' | grep -v grep | awk -v timestamp="$timestamp" -v gpu_load="$gpu_load" -v vram_usage="$vram_usage" '{
                print timestamp "\t" $1 "\t" $2 "\t" $3/1024 "\t" gpu_load "\t" vram_usage "\t" $5  
            }' >> $output_file

            sleep 1
        done
    }
fi


