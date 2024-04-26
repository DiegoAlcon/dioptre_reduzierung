# Expanded Coordinates

def expand_coordinates(x, y):
    # Initialize empty lists for expanded coordinates
    expanded_x = []
    expanded_y = []

    # Iterate through each pair of adjacent coordinates
    for i in range(len(x)):
        # Add the current coordinate to the expanded lists
        expanded_x.append(x[i])
        expanded_y.append(y[i])

        # Interpolate between adjacent coordinates
        dx = x[(i + 1) % len(x)] - x[i]
        dy = y[(i + 1) % len(y)] - y[i]

        # Calculate the number of steps needed for interpolation
        steps = max(abs(dx), abs(dy))

        # Interpolate and add intermediate coordinates
        for j in range(1, steps):
            interp_x = x[i] + (dx * j) // steps
            interp_y = y[i] + (dy * j) // steps
            expanded_x.append(interp_x)
            expanded_y.append(interp_y)

    return expanded_x, expanded_y

# Example usage
x = [1, 1, 3, 3]
y = [0, 2, 2, 0]
expanded_x, expanded_y = expand_coordinates(x, y)
print("Expanded x:", expanded_x)
print("Expanded y:", expanded_y)

