#!/usr/bin/env python3

import subprocess
import matplotlib.pyplot as plt
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the script's directory
os.chdir(script_dir)

# Compile the CUDA program
subprocess.run(['nvcc', '-o', '5_2_2', 'main2.cu'])

# Run the program and capture the output
result = subprocess.run(['./5_2_2'])
""" , capture_output=True, text=True)

# Parse the output
output_lines = result.stdout.splitlines()
sizes = []
global_times = []
shared_times = []
const_times = []

for line in output_lines:
    if line.strip():  # Skip empty lines
        parts = line.split(',')
        sizes.append(int(parts[0]))
        global_times.append(float(parts[1]))
        shared_times.append(float(parts[2]))
        const_times.append(float(parts[3]))

# Convert lists to numpy arrays for easier manipulation
sizes = np.array(sizes)
global_times = np.array(global_times)
shared_times = np.array(shared_times)
const_times = np.array(const_times)

# Create a figure with two subplots
plt.figure(figsize=(14, 6))

# Subplot 1: Linear Scale
plt.subplot(1, 2, 1)
plt.plot(sizes, global_times, marker='o', label='Global Memory Kernel')
plt.plot(sizes, shared_times, marker='s', label='Shared Memory Kernel')
plt.plot(sizes, const_times, marker='x', label='Constant Memory Kernel')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison (Linear Scale)')
plt.legend()
plt.grid(True)

# Subplot 2: Logarithmic Scale
plt.subplot(1, 2, 2)
plt.plot(sizes, global_times, marker='o', label='Global Memory Kernel')
plt.plot(sizes, shared_times, marker='s', label='Shared Memory Kernel')
plt.plot(sizes, const_times, marker='x', label='Constant Memory Kernel')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison (Logarithmic Scale)')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Save the plot as an image
plt.savefig('performance_comparison.png')

# Show the plot
plt.show()
 """