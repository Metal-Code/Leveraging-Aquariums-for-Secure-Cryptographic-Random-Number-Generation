import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import hashlib

# Define paths
data_path = r"D:\Seminar\aquarium_images"  # Update this with your image folder
output_path = "entropy_graphs"
os.makedirs(output_path, exist_ok=True)

def extract_fish_pixels(image, prev_image=None):
    """Extract pixels representing fish by detecting movement between frames"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # If we have a previous image, use motion detection
    fish_mask = None
    if prev_image is not None:
        # Convert both to grayscale for motion detection
        gray_current = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray_current, gray_prev)
        
        # Apply threshold to difference
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        fish_mask = thresh
    else:
        # For first frame, use color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adjust these ranges based on your fish colors
        lower_ranges = [
            np.array([0, 100, 100]),    # Red-Orange
            np.array([100, 100, 100]),  # Blues
            np.array([20, 100, 100])    # Yellows
        ]
        upper_ranges = [
            np.array([20, 255, 255]),
            np.array([140, 255, 255]),
            np.array([40, 255, 255])
        ]
        
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for lower, upper in zip(lower_ranges, upper_ranges):
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        fish_mask = combined_mask
    
    # Apply mask to get fish pixels
    fish_pixels = []
    fish_positions = []
    
    if fish_mask is not None:
        # Find contours of the fish
        contours, _ = cv2.findContours(fish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove small noise artifacts
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Create visualization image
        vis_image = image.copy()
        cv2.drawContours(vis_image, filtered_contours, -1, (0, 255, 0), 2)
        
        # Extract pixels for each fish
        for contour in filtered_contours:
            # Create a mask for this contour
            mask = np.zeros(fish_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Get the centroid of the contour (fish position)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                fish_positions.append((cx, cy))
            
            # Get all pixels in this contour
            y_indices, x_indices = np.where(mask > 0)
            for y, x in zip(y_indices, x_indices):
                fish_pixels.append(image[y, x])
    
    return fish_pixels, fish_positions, vis_image

def calculate_pixel_entropy(pixels, n_bins=256):
    """Calculate entropy of pixel values for cryptographic randomness assessment"""
    if not pixels:
        return 0, 0, 0
    
    pixels_array = np.array(pixels)
    
    # Calculate entropy for each color channel
    entropies = []
    for channel in range(3):  # RGB
        values = pixels_array[:, channel]
        hist, _ = np.histogram(values, bins=n_bins, range=(0, 255))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        channel_entropy = entropy(hist, base=2)
        entropies.append(channel_entropy)
    
    # Average entropy across channels
    return entropies[0], entropies[1], entropies[2]

def calculate_position_entropy(positions, image_shape, grid_size=16):
    """Calculate entropy of fish positions"""
    if not positions:
        return 0
    
    # Create a grid
    grid_counts = np.zeros((grid_size, grid_size))
    
    for x, y in positions:
        x_bin = min(int(x / image_shape[1] * grid_size), grid_size - 1)
        y_bin = min(int(y / image_shape[0] * grid_size), grid_size - 1)
        grid_counts[y_bin, x_bin] += 1
    
    # Flatten and normalize
    grid_counts = grid_counts.flatten()
    if np.sum(grid_counts) > 0:
        grid_probs = grid_counts / np.sum(grid_counts)
        grid_probs = grid_probs[grid_probs > 0]  # Remove zeros
        return entropy(grid_probs, base=2)
    else:
        return 0

def generate_random_bits(pixels, num_bits=256):
    """Generate random bits from fish pixels for cryptographic use"""
    if not pixels:
        return "0" * num_bits, 0
    
    # Convert pixels to bytes
    pixels_array = np.array(pixels, dtype=np.uint8)
    pixels_bytes = pixels_array.tobytes()
    
    # Hash the pixels using SHA-256
    hash_obj = hashlib.sha256(pixels_bytes)
    hash_value = hash_obj.hexdigest()
    
    # Convert to binary
    binary = bin(int(hash_value, 16))[2:].zfill(num_bits)
    
    # Calculate bit entropy (proportion of 1s should be close to 0.5 for good randomness)
    ones_count = binary.count('1')
    bit_entropy = abs(0.5 - (ones_count / num_bits)) * 2  # Normalized deviation from 0.5
    
    return binary[:num_bits], 1 - bit_entropy  # Higher value means better randomness

def create_entropy_visualizations(data_df, random_bits, output_path):
    """Generate comprehensive visualizations for entropy analysis"""
    
    # Set a professional style for the plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. TIME SERIES ENTROPY GRAPH
    plt.figure(figsize=(14, 8))
    plt.plot(data_df['frame'], data_df['r_entropy'], 'r-', label='Red Channel', linewidth=2)
    plt.plot(data_df['frame'], data_df['g_entropy'], 'g-', label='Green Channel', linewidth=2)
    plt.plot(data_df['frame'], data_df['b_entropy'], 'b-', label='Blue Channel', linewidth=2)
    plt.plot(data_df['frame'], data_df['position_entropy'], 'k-', label='Position', linewidth=2)
    plt.plot(data_df['frame'], data_df['combined_entropy'], 'm-', label='Combined', linewidth=3)
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.title('Entropy Measures Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add theoretical maximum line
    plt.axhline(y=8, color='gray', linestyle='--', alpha=0.7, label='Theoretical Max (8 bits)')
    
    # Add annotations for significant points
    max_entropy_frame = data_df['combined_entropy'].idxmax()
    plt.scatter(data_df['frame'][max_entropy_frame], data_df['combined_entropy'][max_entropy_frame], 
                color='red', s=100, zorder=5)
    plt.annotate(f'Max: {data_df["combined_entropy"][max_entropy_frame]:.2f} bits', 
                 (data_df['frame'][max_entropy_frame], data_df['combined_entropy'][max_entropy_frame]),
                 xytext=(10, -30), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'entropy_time_series.png'), dpi=300)
    plt.close()
    
    # 2. ENTROPY DISTRIBUTION HISTOGRAM
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data_df['r_entropy'], kde=True, color='red', bins=15)
    plt.title('Red Channel Entropy Distribution', fontsize=14)
    plt.xlabel('Entropy (bits)', fontsize=12)
    
    plt.subplot(2, 2, 2)
    sns.histplot(data_df['g_entropy'], kde=True, color='green', bins=15)
    plt.title('Green Channel Entropy Distribution', fontsize=14)
    plt.xlabel('Entropy (bits)', fontsize=12)
    
    plt.subplot(2, 2, 3)
    sns.histplot(data_df['b_entropy'], kde=True, color='blue', bins=15)
    plt.title('Blue Channel Entropy Distribution', fontsize=14)
    plt.xlabel('Entropy (bits)', fontsize=12)
    
    plt.subplot(2, 2, 4)
    sns.histplot(data_df['position_entropy'], kde=True, color='purple', bins=15)
    plt.title('Position Entropy Distribution', fontsize=14)
    plt.xlabel('Entropy (bits)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'entropy_distributions.png'), dpi=300)
    plt.close()
    
    # 3. ENTROPY HEATMAP (CORRELATION MATRIX)
    plt.figure(figsize=(10, 8))
    correlation_columns = ['r_entropy', 'g_entropy', 'b_entropy', 'position_entropy', 
                        'combined_entropy', 'randomness_quality']
    
    # Create correlation matrix
    correlation_matrix = data_df[correlation_columns].corr()
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create the heatmap with annotations
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, fmt='.2f')
    
    plt.title('Correlation Between Different Entropy Measures', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'entropy_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 4. SPATIAL ENTROPY VISUALIZATION
    # Create a grid to visualize where fish are most commonly found
    grid_size = 16
    grid_counts = np.zeros((grid_size, grid_size))
    
    # Accumulate fish positions across frames
    for _, row in data_df.iterrows():
        # Use the frame number to access the corresponding image
        frame_num = int(row['frame'])
        if frame_num <= len(os.listdir(data_path)):
            img_file = sorted([f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))])[frame_num-1]
            img_path = os.path.join(data_path, img_file)
            
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get fish positions from the previous processing
                _, positions, _ = extract_fish_pixels(image)
                
                # Add to grid
                for x, y in positions:
                    x_bin = min(int(x / image.shape[1] * grid_size), grid_size - 1)
                    y_bin = min(int(y / image.shape[0] * grid_size), grid_size - 1)
                    grid_counts[y_bin, x_bin] += 1
    
    # Create a spatial entropy visualization
    plt.figure(figsize=(10, 8))
    mask = grid_counts == 0  # For masking zeros in the visualization
    
    # Add a custom colormap for better visualization
    colors = ["darkblue", "blue", "lightblue", "lightgreen", "yellow", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    sns.heatmap(grid_counts, cmap=cmap, mask=mask, square=True, cbar_kws={'label': 'Fish Presence Count'})
    plt.title('Spatial Distribution of Fish (Cumulative)', fontsize=16)
    plt.xlabel('X Position (binned)', fontsize=12)
    plt.ylabel('Y Position (binned)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'spatial_entropy_heatmap.png'), dpi=300)
    plt.close()
    
    # 5. 3D SCATTER PLOT OF RGB ENTROPY
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot with points colored by combined entropy
    scatter = ax.scatter(data_df['r_entropy'], data_df['g_entropy'], data_df['b_entropy'], 
                       c=data_df['combined_entropy'], cmap='plasma', s=50, alpha=0.8)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Combined Entropy (bits)', fontsize=12)
    
    # Add titles and labels
    ax.set_title('3D Visualization of RGB Channel Entropy', fontsize=16)
    ax.set_xlabel('Red Channel Entropy (bits)', fontsize=12)
    ax.set_ylabel('Green Channel Entropy (bits)', fontsize=12)
    ax.set_zlabel('Blue Channel Entropy (bits)', fontsize=12)
    
    # Add grid
    ax.grid(True)
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'rgb_entropy_3d.png'), dpi=300)
    plt.close()
    
    # 6. BIT VISUALIZATION FOR CRYPTOGRAPHIC ANALYSIS
    if random_bits:
        plt.figure(figsize=(15, 12))
        
        # Take some sample frames for bit visualization
        sample_indices = [0, len(random_bits)//3, (2*len(random_bits))//3, len(random_bits)-1]
        sample_indices = [i for i in sample_indices if i < len(random_bits)]
        
        for i, idx in enumerate(sample_indices):
            bits = random_bits[idx]
            bit_array = np.array([int(bit) for bit in bits[:256]])  # Use first 256 bits
            
            # Reshape to a square
            bit_square = bit_array.reshape(16, 16)
            
            plt.subplot(2, 2, i+1)
            plt.imshow(bit_square, cmap='binary', interpolation='none')
            
            # Calculate bit stats
            ones_ratio = np.sum(bit_square) / bit_square.size
            entropy_val = entropy([ones_ratio, 1-ones_ratio], base=2) if 0 < ones_ratio < 1 else 0
            
            plt.title(f'Frame {sample_indices[i]+1} Random Bits\n'
                     f'1s: {ones_ratio:.2%}, Entropy: {entropy_val:.2f} bits', 
                     fontsize=12)
            
            plt.colorbar(ticks=[0, 1], label='Bit Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'random_bit_visualization.png'), dpi=300)
        plt.close()
    
    # 7. RANDOMNESS QUALITY GAUGE CHART
    def create_gauge_chart(value, title, min_val=0, max_val=1, threshold=0.9):
        # Create a half-circle gauge chart
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'polar'})
        
        # Define the angles for the half-circle
        theta = np.linspace(0, np.pi, 100)
        
        # Create the colored background
        radii = np.ones_like(theta)
        width = np.pi / len(theta)
        
        # Calculate normalized value
        norm_value = (value - min_val) / (max_val - min_val)
        
        # Define colors for different sections
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(theta)))
        
        # Draw the background
        ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.8)
        
        # Add a needle to indicate the value
        needle_theta = np.pi * norm_value
        ax.plot([0, needle_theta], [0, 0.9], 'k-', linewidth=3)
        ax.plot([0, needle_theta], [0, 0.9], 'w-', linewidth=1)
        
        # Add a circle at the base of the needle
        ax.plot(0, 0, 'o', markersize=10, color='black')
        ax.plot(0, 0, 'o', markersize=8, color='white')
        
        # Add threshold line
        threshold_theta = np.pi * (threshold - min_val) / (max_val - min_val)
        ax.plot([threshold_theta, threshold_theta], [0.7, 1.0], 'k--', linewidth=1.5)
        
        # Set the title
        ax.text(np.pi/2, 1.2, title, fontsize=16, ha='center')
        
        # Add the value
        ax.text(np.pi/2, 0.5, f"{value:.2f}", fontsize=20, ha='center', fontweight='bold')
        
        # Add labels
        ax.text(0, 0.3, f"{min_val}", fontsize=12, ha='left')
        ax.text(np.pi, 0.3, f"{max_val}", fontsize=12, ha='right')
        ax.text(threshold_theta, 1.1, "Threshold", fontsize=10, ha='center')
        
        # Customize the appearance
        ax.set_rmax(1.5)
        ax.set_rticks([])  # Remove radial ticks
        ax.set_xticks([])  # Remove angular ticks
        ax.spines['polar'].set_visible(False)
        
        return fig
    
    # Create a gauge chart for average randomness quality
    avg_quality = data_df['randomness_quality'].mean()
    gauge_fig = create_gauge_chart(avg_quality, "Average Randomness Quality")
    gauge_fig.savefig(os.path.join(output_path, 'randomness_quality_gauge.png'), dpi=300, bbox_inches='tight')
    plt.close(gauge_fig)
    
    # 8. ENTROPY OVER TIME WITH POLYNOMIAL TREND
    plt.figure(figsize=(14, 8))
    
    # Plot the entropy data
    plt.scatter(data_df['frame'], data_df['combined_entropy'], color='blue', alpha=0.6, label='Combined Entropy')
    
    # Add polynomial trend line
    z = np.polyfit(data_df['frame'], data_df['combined_entropy'], 3)
    p = np.poly1d(z)
    x_trend = np.linspace(data_df['frame'].min(), data_df['frame'].max(), 100)
    y_trend = p(x_trend)
    
    plt.plot(x_trend, y_trend, 'r-', linewidth=2, label='Trend Line')
    
    # Add rolling average
    window_size = max(3, len(data_df) // 10)  # At least 3, or 10% of the data
    rolling_avg = data_df['combined_entropy'].rolling(window=window_size, center=True).mean()
    plt.plot(data_df['frame'], rolling_avg, 'g--', linewidth=2, label=f'{window_size}-Frame Moving Average')
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Combined Entropy (bits)', fontsize=12)
    plt.title('Entropy Trend Analysis', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for the trend
    if y_trend[-1] > y_trend[0]:
        trend_text = "Increasing Trend: Entropy grows as simulation progresses"
    else:
        trend_text = "Decreasing Trend: Entropy declines as simulation progresses"
    
    plt.annotate(trend_text, xy=(0.5, 0.02), xycoords='figure fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'entropy_trend_analysis.png'), dpi=300)
    plt.close()
    
    # 9. ENTROPY VS FISH COUNT SCATTER PLOT
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with size proportional to randomness quality
    sizes = data_df['randomness_quality'] * 100 + 20
    scatter = plt.scatter(data_df['fish_count'], data_df['combined_entropy'], 
                         c=data_df['frame'], cmap='viridis', 
                         s=sizes, alpha=0.7)
    
    # Add a colorbar for the frame number
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frame Number', fontsize=12)
    
    # Add a best-fit line
    if len(data_df) > 1:  # Ensure we have enough data points
        z = np.polyfit(data_df['fish_count'], data_df['combined_entropy'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data_df['fish_count'].min(), data_df['fish_count'].max(), 100)
        plt.plot(x_trend, p(x_trend), 'r-', linewidth=2)
        
        # Calculate and display correlation
        corr = data_df['fish_count'].corr(data_df['combined_entropy'])
        plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                    fontsize=12)
    
    plt.xlabel('Number of Fish Detected', fontsize=12)
    plt.ylabel('Combined Entropy (bits)', fontsize=12)
    plt.title('Relationship Between Fish Count and Entropy', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fish_count_vs_entropy.png'), dpi=300)
    plt.close()

def process_images(data_path, output_path):
    """Process all images in the dataset and generate entropy data"""
    # Get sorted list of images
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print("No image files found in the specified directory.")
        return None, None
    
    # Initialize results storage
    results = {
        'frame': [],
        'r_entropy': [],
        'g_entropy': [],
        'b_entropy': [],
        'position_entropy': [],
        'randomness_quality': [],
        'fish_count': [],
        'pixel_count': []
    }
    
    # Store random bits from each frame
    all_random_bits = []
    
    # Process images
    prev_image = None
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        
        # Read image
        img_path = os.path.join(data_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract fish pixels and positions
        fish_pixels, fish_positions, vis_image = extract_fish_pixels(image, prev_image)
        
        # Calculate entropy values
        r_entropy, g_entropy, b_entropy = calculate_pixel_entropy(fish_pixels)
        position_entropy = calculate_position_entropy(fish_positions, image.shape)
        
        # Generate random bits
        binary_string, randomness_quality = generate_random_bits(fish_pixels)
        all_random_bits.append(binary_string)
        
        # Store results
        results['frame'].append(i+1)
        results['r_entropy'].append(r_entropy)
        results['g_entropy'].append(g_entropy)
        results['b_entropy'].append(b_entropy)
        results['position_entropy'].append(position_entropy)
        results['randomness_quality'].append(randomness_quality)
        results['fish_count'].append(len(fish_positions))
        results['pixel_count'].append(len(fish_pixels))
        
        # Save visualization image
        cv2.imwrite(os.path.join(output_path, f"fish_detection_{i+1:03d}.jpg"), 
                   cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Set current image as previous for next iteration
        prev_image = image
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate combined entropy
    df['combined_entropy'] = (df['r_entropy'] + df['g_entropy'] + df['b_entropy'] + df['position_entropy']) / 4
    
    # Save to CSV for future reference
    df.to_csv(os.path.join(output_path, 'entropy_data.csv'), index=False)
    
    return df, all_random_bits

def main():
    """Main function to process images and generate entropy analysis graphs"""
    print("Starting fish movement entropy analysis...")
    
    # Process all images and get entropy data
    df, random_bits = process_images(data_path, output_path)
    
    if df is None:
        print("Error: No valid data to analyze.")
        return
    
    print(f"Processed {len(df)} images. Generating visualizations...")
    
    # Generate all visualizations
    create_entropy_visualizations(df, random_bits, output_path)
    
    print(f"Analysis complete! All graphs have been saved to {output_path}")
    print("\nEntropy Statistics Summary:")
    print(f"Average Combined Entropy: {df['combined_entropy'].mean():.4f} bits")
    print(f"Maximum Combined Entropy: {df['combined_entropy'].max():.4f} bits")
    print(f"Average Randomness Quality: {df['randomness_quality'].mean():.4f}")
    print(f"Average Fish Count: {df['fish_count'].mean():.2f}")

if __name__ == "__main__":
    main()