import numpy as np

def cauchy_argument_principle(x, y, px, py):
    n = len(x)
    winding_number = 0

    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[(i + 1) % n], y[(i + 1) % n]

        if (yi <= py < yj or yi <= py < yi) and px <= max(xi, xj):
            intersection_x = (py - yi) * (xj - xi) / (yj - yi) + xi

            if px < intersection_x:
                winding_number += 1

    return winding_number % 2 == 1

x = [1, 4, 5, 3]
y = [1, 2, 5, 4]

x_expanded = []
y_expanded = []

for px in range(1,6):
    for py in range(1,6):
        is_inside = cauchy_argument_principle(x, y, px, py)
        if is_inside:
            x_expanded.append(px)
            y_expanded.append(py)
        print("Is the point inside the contour? ", is_inside)