#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import subprocess
import csv

# Compile the CUDA program
subprocess.run(['nvcc', '-o', 'performance_test', 'flip3.cu'])

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
cpuTimes = data[:, 1]
gpuKernelTimes = data[:, 2]
gpuTotalTimes = data[:, 3]

# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(sizes, cpuTimes, label='CPU')
plt.loglog(sizes, gpuKernelTimes, label='GPU Kernel')
plt.loglog(sizes, gpuTotalTimes, label='GPU Total')

plt.xlabel('Array Size')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Save plot to file
plt.savefig('performance_comparison.png')