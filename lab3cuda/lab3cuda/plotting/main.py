#!/usr/bin/env python3

import subprocess
from PIL import Image
import matplotlib.pyplot as plt

# Compile the CUDA program
subprocess.run(['nvcc', '-o', 'im_test.exe', 'main.cu'], check=True)

# Run the program and capture the output
result = subprocess.run(['im_test.exe'], capture_output=True, text=True)

# Open the PPM image
image = Image.open('output_image.ppm')

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.title('Output Image')
plt.show()

# Parse the output
run_times = [float(line) for line in result.stdout.strip().split('\n')]
runOne, runThree = run_times

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(['With Thread Divergence', 'Without Thread Divergence'], [runOne, runThree])

# Customize the plot
plt.title('Comparison of GPU Filter Run Times')
plt.ylabel('Runtime (seconds)')
plt.ylim(bottom=0)  # Start y-axis from 0

# Add grid for better readability
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add value labels on top of each bar
for i, v in enumerate([runOne, runThree]):
    plt.text(i, v, f'{v:.6e}', ha='center', va='bottom')

# Adjust layout and display
plt.tight_layout()
plt.show()
