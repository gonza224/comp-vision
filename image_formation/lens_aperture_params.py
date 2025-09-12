import matplotlib.pyplot as plt
import numpy as np

from utils import aperture_diameter, thin_lens_zi


def lens_to_image_distance_plot():
    # The focal lengths in mm
    focal_lengths = [3, 9, 50, 200]

    # The z0 range: from 1.1*f to 10^4 mm
    z0_min = 1.1 * min(focal_lengths)
    z0_max = 10000
    print(f"z0_min: {z0_min}, z0_max: {z0_max}")

    # Use 4 points per mm
    step = 0.25  # mm  (1/4 mm = 4 points per mm)
    z0_points = []
    z = z0_min
    while z <= z0_max:
        z0_points.append(z)
        z += step

    z0_points = np.array(z0_points)

    plt.figure(figsize=(10, 6))

    # Plot zi vs z0 for each focal length
    colors = ['r', 'g', 'b', 'y']
    for f in focal_lengths:
        zi_values = thin_lens_zi(f, z0_points)

        # This is so the curve line doesnt go over the dashed line
        bad = z0_points <= f * 1.001
        zi_values = zi_values.astype(float)
        zi_values[bad] = np.nan
        plt.loglog(z0_points, zi_values, label=f"f = {f} mm", color=colors[focal_lengths.index(f)])

        # Vertical dashed line at z0 = f
        plt.axvline(f, color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)

    # Format plot
    plt.xlabel("Object Distance z_0 (mm)")
    plt.ylabel("Image Distance z_i (mm)")
    plt.ylim(0, 3000)
    plt.title("lens-to-image distance (z_i) as a function of the object distance (z_0) for four different focal lengths")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


def aperture_diameter_plot():
    # Define focal-length domain (in mm)
    f_min = 10
    f_max = 600
    num_points = 1000
    f_values = np.linspace(f_min, f_max, num_points)

    # Popular f-numbers to plot https://www.reddit.com/r/photography/comments/yxwasv/what_f_stops_are_you_favorites/
    f_numbers = [1.4, 1.8, 3.2, 4, 5.6, 6.3, 7.1, 8, 11, 13, 16]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot D = f / N for each f-number
    for N in f_numbers:
        D_values = aperture_diameter(f_values, N)
        plt.plot(f_values, D_values, label=f"f/{N:g}")

    plt.xlabel("Focal length f (mm)")
    plt.ylabel("Aperture diameter D (mm)")
    plt.title("Aperture Diameter vs Focal Length for Popular f-Numbers")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="f-number")
    plt.tight_layout()
    plt.show()

def real_world_aperture_diameters():
    lenses = [(24,   1.4),
        (50, 1.8),
        (70, 2.8),
        (200, 2.8),
        (400, 2.8),
        (600, 4.0),
    ]

    for lens in lenses:
        f, N = lens
        D = aperture_diameter(f, N)
        print(f"Aperture diameter for {f}mm f/{N}: {D:.2f}mm")

if __name__ == "__main__":
    # lens_to_image_distance_plot()
    # aperture_diameter_plot()
    real_world_aperture_diameters()
