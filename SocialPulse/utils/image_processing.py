import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the input signature image.
    
    Args:
        image: Input image (OpenCV format)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Check if image is grayscale or color
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours to identify the signature region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the signature)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding around the signature
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        # Crop the signature region
        signature = binary[y:y+h, x:x+w]
        
        # Resize to the target size while maintaining aspect ratio
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            # Width is greater than height
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is greater than width
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
        
        resized = cv2.resize(signature, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a blank canvas of the target size
        canvas = np.zeros(target_size, dtype=np.uint8)
        
        # Calculate position to paste the resized image (center)
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        
        # Paste the resized image on the canvas
        canvas[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized
        
        return canvas
    
    # If no contours found, just resize and return the binary image
    return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """
    Normalize the preprocessed image.
    
    Args:
        image: Preprocessed binary image
        
    Returns:
        Normalized image
    """
    # Ensure the image is in the correct format (float32 between 0 and 1)
    normalized = image.astype(np.float32) / 255.0
    
    return normalized

def augment_image(image):
    """
    Apply data augmentation to generate variations of the signature.
    
    Args:
        image: Input image
        
    Returns:
        List of augmented images
    """
    augmented_images = []
    original = image.copy()
    
    # Add the original image
    augmented_images.append(original)
    
    # Slight rotation variations
    for angle in [-5, -2, 2, 5]:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        augmented_images.append(rotated)
    
    # Slight scaling variations
    for scale in [0.95, 1.05]:
        h, w = image.shape[:2]
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas
        canvas = np.zeros((h, w), dtype=np.float32)
        
        # Calculate position to paste the scaled image (center)
        new_h, new_w = scaled.shape[:2]
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        
        # Ensure the paste coordinates are valid
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
        
        # Adjust dimensions if the scaled image is larger than the canvas
        paste_w = min(new_w, w - paste_x)
        paste_h = min(new_h, h - paste_y)
        
        # Paste the scaled image on the canvas
        canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = scaled[:paste_h, :paste_w]
        
        augmented_images.append(canvas)
    
    return augmented_images
