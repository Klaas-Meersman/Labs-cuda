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

import matplotlib.pyplot as plt
import numpy as np

# Load data from the CSV file
data = np.loadtxt('performance_data.csv', delimiter=',')

# Extract columns
sizes = data[:, 0]
cpu_simple_times = data[:, 1]
cpu_nested_times = data[:, 2]
gpu_atomic_times = data[:, 3]
gpu_reduction_times = data[:, 4]

# Plotting
plt.figure(figsize=(12, 8))
plt.loglog(sizes, cpu_simple_times, label='CPU Simple', marker='o')
plt.loglog(sizes, cpu_nested_times, label='CPU Nested/Reduction', marker='s')
plt.loglog(sizes, gpu_atomic_times, label='GPU Atomic', marker='^')
plt.loglog(sizes, gpu_reduction_times, label='GPU Reduction', marker='D')

plt.xlabel('Array Size')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Save plot to file
plt.savefig('performance_comparison.png')