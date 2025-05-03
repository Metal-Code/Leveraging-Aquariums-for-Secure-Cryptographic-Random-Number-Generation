import cv2
import numpy as np
import os
import random

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def generate_gradient_background(width, height):
    """Create a vertical gradient simulating water depth."""
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        shade = 200 - int((y / height) * 80)
        base[y, :] = (shade + random.randint(0, 10), shade + 30, shade + 50)
    return base

def add_lighting_effects(image):
    """Overlay caustic lighting effect to mimic underwater lighting."""
    light = np.zeros_like(image, dtype=np.uint8)
    rows, cols, _ = image.shape
    for i in range(20):
        center = (random.randint(0, cols), random.randint(0, rows))
        radius = random.randint(50, 150)
        intensity = random.randint(20, 50)
        cv2.circle(light, center, radius, (intensity, intensity, intensity), -1)
    return cv2.addWeighted(image, 1, light, 0.3, 0)

def generate_synthetic_aquarium_images(folder, num_images=100, width=640, height=480, num_fish=10):
    create_folder(folder)
    
    # Initialize fish attributes
    fish_positions = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_fish)]
    fish_velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(num_fish)]
    fish_colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_fish)]
    fish_sizes = [(random.randint(8, 20), random.randint(5, 15)) for _ in range(num_fish)]  # (major_axis, minor_axis)

    for i in range(num_images):
        # Create gradient background
        image = generate_gradient_background(width, height)

        # Add lighting effects
        if i % 10 == 0:
            image = add_lighting_effects(image)

        # Move fish with velocity vectors
        new_positions = []
        for idx, (x, y) in enumerate(fish_positions):
            vx, vy = fish_velocities[idx]
            x = (x + vx) % width
            y = (y + vy) % height
            new_positions.append((int(x), int(y)))
        fish_positions = new_positions

        # Draw fish (as ellipses)
        for idx, (x, y) in enumerate(fish_positions):
            color = fish_colors[idx]
            axes = fish_sizes[idx]
            angle = random.randint(0, 360)
            cv2.ellipse(image, (x, y), axes, angle, 0, 360, color, -1)

        # Draw bubbles (white with light opacity simulation)
        for _ in range(5):
            bx = random.randint(0, width)
            by = random.randint(0, height)
            br = random.randint(2, 6)
            overlay = image.copy()
            cv2.circle(overlay, (bx, by), br, (255, 255, 255), -1)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

        # Save image
        filename = os.path.join(folder, f"frame_{i:03d}.png")
        cv2.imwrite(filename, image)

if __name__ == "__main__":
    output_folder = r"D:\test"  # Adjust if needed
    generate_synthetic_aquarium_images(output_folder, num_images=100, width=640, height=480, num_fish=12)
