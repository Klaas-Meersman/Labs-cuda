#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import csv

# Compile the CUDA program
subprocess.run(['nvcc', '-o', 'performance_test', 'maxArray.cu'])

# Run the program and capture the output
output = subprocess.run(['./performance_test'], capture_output=True, text=True)

# Save the output to a CSV file
with open('performance_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for line in output.stdout.split('\n'):
        if line:  # Skip empty lines
            writer.writerow(line.split(','))

# Load data from the CSV file
data = np.loadtxt('performance_data.csv', delimiter=',')

# Extract columns
sizes = data[:, 0]
cpu_simple_times = data[:, 1]
cpu_nested_times = data[:, 2]
gpu_atomic_times = data[:, 3]
gpu_reduction_times = data[:, 4]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot logarithmic scale
ax1.loglog(sizes, cpu_simple_times, label='CPU Simple', marker='o')
ax1.loglog(sizes, cpu_nested_times, label='CPU Nested/Reduction', marker='s')
ax1.loglog(sizes, gpu_atomic_times, label='GPU Atomic', marker='^')
ax1.loglog(sizes, gpu_reduction_times, label='GPU Reduction', marker='D')
ax1.set_xlabel('Array Size')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Performance Comparison (Log-Log Scale)')
ax1.legend()
ax1.grid(True)

# Plot with logarithmic x-axis and linear y-axis
ax2.semilogx(sizes, cpu_simple_times, label='CPU Simple', marker='o')
ax2.semilogx(sizes, cpu_nested_times, label='CPU Nested/Reduction', marker='s')
ax2.semilogx(sizes, gpu_atomic_times, label='GPU Atomic', marker='^')
ax2.semilogx(sizes, gpu_reduction_times, label='GPU Reduction', marker='D')
ax2.set_xlabel('Array Size')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Performance Comparison (Log-Linear Scale)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('performance_comparison_combined.png')
plt.close()

print("Combined plot has been saved as 'performance_comparison_combined.png'")