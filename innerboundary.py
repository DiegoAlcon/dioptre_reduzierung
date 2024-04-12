#def expand_boundary(x, y):
    # Create a set to store all unique pixel coordinates within the boundary
#    pixel_set = set()

    # Iterate through each pair of adjacent boundary points
#    for i in range(len(x)):
#        x1, y1 = x[i], y[i]
#        x2, y2 = x[(i + 1) % len(x)]  # Wrap around to the first point for the last pair

        # Determine the minimum and maximum x-coordinates between the two points
#        min_x, max_x = min(x1, x2), max(x1, x2)

        # Add all pixel coordinates within the range [min_x, max_x] to the set
#        for px in range(min_x, max_x + 1):
            # Calculate the corresponding y-coordinate using linear interpolation
#            py = y1 + (y2 - y1) * (px - x1) / (x2 - x1)
#            pixel_set.add((px, int(py)))  # Convert y-coordinate to integer

    # Extract the expanded x and y coordinates from the set
#    expanded_x, expanded_y = zip(*pixel_set)

#    return list(expanded_x), list(expanded_y)

# Example usage
#x = [1, 1, 1, 2, 3, 3, 3, 2]
#y = [0, 1, 2, 2, 2, 1, 0, 0]
#expanded_x, expanded_y = expand_boundary(x, y)
#print("Expanded x:", expanded_x)
#print("Expanded y:", expanded_y)

def expand_boundary(x, y):
    # Create a set to store all pixel coordinates within the boundary
    pixel_set = set()

    # Iterate through each pair of adjacent boundary points
    for i in range(len(x)):
        x1, y1 = x[i], y[i]
        x2, y2 = x[(i + 1) % len(x)], y[(i + 1) % len(y)]

        # Determine the minimum and maximum x-coordinates between the two points
        min_x = min(x1, x2)
        max_x = max(x1, x2)

        # Add all pixel coordinates within the x-range to the set
        for px in range(min_x, max_x + 1):
            pixel_set.add((px, y1))

        # Determine the minimum and maximum y-coordinates between the two points
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        # Add all pixel coordinates within the y-range to the set
        for py in range(min_y, max_y + 1):
            pixel_set.add((x1, py))

    # Extract the expanded x and y coordinates from the set
    expanded_x, expanded_y = zip(*pixel_set)

    return list(expanded_x), list(expanded_y)

# Example usage:
x = [1, 1, 1, 2, 3, 3, 3, 2]
y = [0, 1, 2, 2, 2, 1, 0, 0]
expanded_x, expanded_y = expand_boundary(x, y)
print("Expanded x:", expanded_x)
print("Expanded y:", expanded_y)

