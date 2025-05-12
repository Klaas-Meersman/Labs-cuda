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
subprocess.run(['nvcc', '-o', '5_1_3', 'main3.cu'], check=True)

# Run the program and capture the output
result = subprocess.run(['5_1_3.exe'], capture_output=True, text=True)

# Open the PPM image
#image = Image.open('output_image.ppm')

# Display the image
#plt.figure(figsize=(10, 10))
#plt.imshow(image)
#plt.axis('off')  # Turn off axis
#plt.title('Output Image')
#plt.show()

# Parse the output
data = [line.strip().split(',') for line in result.stdout.strip().split('\n')]
gridsizes = [int(row[0]) for row in data]
median_uncoalesced = [float(row[1]) for row in data]
median_coalesced = [float(row[2]) for row in data]



plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(gridsizes, median_uncoalesced, marker='o', linestyle='-', label='Median Uncoalesced')
plt.plot(gridsizes, median_coalesced, marker='s', linestyle='--', label='Median Coalesced')

# Add labels and title
plt.xlabel('Grid size')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Median Uncoalesced and Coalesced Times')
plt.legend()  # Show legend
plt.grid(True)  # Add gridlines

# Show the plot
plt.show()
