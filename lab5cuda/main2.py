#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pandas as pd
from io import StringIO

# Compile the CUDA program
try:
    compile_result = subprocess.run(['nvcc', '-o', 'exeD', 'reduction.cu'], check=True, capture_output=True, text=True)
    print("Compilation successful.")
except subprocess.CalledProcessError as e:
    print(f"Compilation failed: {e.stderr}")
    exit()

# Run the program and capture the output
try:
    run_result = subprocess.run(['./exeD'], capture_output=True, text=True, check=True)
    output = run_result.stdout
    print("Execution successful.")
except subprocess.CalledProcessError as e:
    print(f"Execution failed: {e.stderr}")
    exit()

# Parse the output into a DataFrame
csv_data = StringIO(output)
df = pd.read_csv(csv_data, header=None, names=['Size', 'Sync_Total_Time', 'Async_Total_Time'])

# Print the DataFrame
print("\nCaptured Data:")
print(df.to_string(index=False))

# Convert DataFrame columns to NumPy arrays
size = df['Size'].to_numpy()
sync_total_time = df['Sync_Total_Time'].to_numpy()
async_total_time = df['Async_Total_Time'].to_numpy()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(size, sync_total_time, label='Sync Total', marker='s')
plt.plot(size, async_total_time, label='Async Total', marker='D')

plt.xlabel('Array Size')
plt.ylabel('Time (seconds)')
plt.xscale('log')
plt.yscale('log')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.close()

print("\nPerformance comparison plot saved as 'performance_comparison.png'")
