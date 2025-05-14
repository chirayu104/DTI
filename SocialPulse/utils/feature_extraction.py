import numpy as np
import cv2
from skimage.feature import hog
from skimage import feature

def extract_features(image):
    """
    Extract features from the preprocessed signature image.
    Uses a combination of HOG features and LBP texture features.
    
    Args:
        image: Preprocessed and normalized image
        
    Returns:
        Feature vector
    """
    # Make sure image is the right type
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # If image is 2D, make sure it's in the range [0, 1]
    if len(image.shape) == 2:
        if image.max() > 1.0:
            image = image / 255.0
    
    # 1. Extract HOG (Histogram of Oriented Gradients) features
    hog_features = extract_hog_features(image)
    
    # 2. Extract LBP (Local Binary Pattern) features for texture
    lbp_features = extract_lbp_features(image)
    
    # 3. Extract simple statistical features
    stat_features = extract_statistical_features(image)
    
    # 4. Extract contour-based features
    contour_features = extract_contour_features(image)
    
    # Combine all features into a single feature vector
    combined_features = np.concatenate([
        hog_features, 
        lbp_features, 
        stat_features,
        contour_features
    ])
    
    return combined_features

def extract_hog_features(image, pixels_per_cell=(8, 8)):
    """
    Extract HOG features from the image
    
    Args:
        image: Input image
        pixels_per_cell: Cell size for HOG
        
    Returns:
        HOG feature vector
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Convert to uint8 if needed for HOG
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Calculate HOG features
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features

def extract_lbp_features(image, points=24, radius=3):
    """
    Extract Local Binary Pattern features for texture analysis
    
    Args:
        image: Input image
        points: Number of points in LBP calculation
        radius: Radius for LBP
        
    Returns:
        LBP histogram features
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Calculate LBP
    lbp = feature.local_binary_pattern(image, points, radius, method='uniform')
    
    # Calculate the histogram of LBP
    n_bins = points + 2  # For uniform LBP
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def extract_statistical_features(image):
    """
    Extract basic statistical features from the image
    
    Args:
        image: Input image
        
    Returns:
        Statistical features vector
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate basic statistics
    mean = np.mean(image)
    std = np.std(image)
    
    # Calculate histogram
    hist, _ = np.histogram(image.ravel(), bins=10, range=(0, 1), density=True)
    
    # Calculate image moments
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Combine all statistical features
    stat_features = np.concatenate([[mean, std], hist, hu_moments])
    
    return stat_features

def extract_contour_features(image):
    """
    Extract features based on contour properties
    
    Args:
        image: Input image
        
    Returns:
        Contour-based features
    """
    # Convert to binary image if not already
    if image.dtype != np.uint8:
        binary = (image * 255).astype(np.uint8)
    else:
        binary = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Return zeros if no contours found
        return np.zeros(5)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate contour properties
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate contour complexity (ratio of perimeter to area)
    complexity = perimeter / (area + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Calculate contour bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / (h + 1e-10)
    
    # Calculate contour density (area of contour / area of bounding rectangle)
    rect_area = w * h
    density = area / (rect_area + 1e-10)
    
    return np.array([area, perimeter, complexity, aspect_ratio, density])
