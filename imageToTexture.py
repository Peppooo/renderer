from PIL import Image

# Load the image
img = Image.open(input("nome file: "))

# Convert image to RGB (optional, ensures 3 channels)
img = img.convert("RGB")

# Get image size
width, height = img.size

data = []

# Loop through each pixel
for y in range(height):
    for x in range(width):
        r, g, b = img.getpixel((x, y))
        # Do something with the pixel values
        print(f"Pixel at ({x},{y}): R={r}, G={g}, B={b}")
        data.append(r)
        data.append(g)
        data.append(b)

data = bytes(data)


# Save to a binary file
with open("out.tex", "wb") as f:
    f.write(data)