#!/usr/bin/env python3

import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the script's directory
os.chdir(script_dir)

# Compile the CUDA program
subprocess.run(['nvcc', '-o', '5_1_1', 'main.cu'], check=True)

# Run the program and capture the output
result = subprocess.run(['5_1_1.exe'], capture_output=True, text=True)

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

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(['Uncoalesced Access', 'Coalesced Access'], [runOne, runThree])

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
