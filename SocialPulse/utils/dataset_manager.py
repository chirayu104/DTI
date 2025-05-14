import numpy as np
import uuid
import random
from sklearn.model_selection import train_test_split

class DatasetManager:
    """
    Manages the dataset of reference signatures for the signature verification system.
    Handles storage, retrieval, and preparation of signature data for model training.
    """
    
    def __init__(self):
        """Initialize the dataset manager"""
        # Dictionary to store reference signatures
        # Structure: {user_id: [feature_vectors]}
        self.reference_signatures = {}
        
        # Dictionary to store user names mapped to user IDs
        # Structure: {user_name: user_id}
        self.user_name_to_id = {}
        
    def add_reference_signature(self, user_name, feature_vector):
        """
        Add a reference signature for a user
        
        Args:
            user_name: Name of the user
            feature_vector: Feature vector of the signature
            
        Returns:
            None
        """
        # If user doesn't exist, create a new user ID
        if user_name not in self.user_name_to_id:
            user_id = str(uuid.uuid4())
            self.user_name_to_id[user_name] = user_id
            self.reference_signatures[user_id] = []
        
        # Get the user ID
        user_id = self.user_name_to_id[user_name]
        
        # Add the feature vector to the user's reference signatures
        self.reference_signatures[user_id].append(feature_vector)
    
    def get_user_signatures(self, user_name):
        """
        Get all reference signatures for a user
        
        Args:
            user_name: Name of the user
            
        Returns:
            List of feature vectors
        """
        if user_name not in self.user_name_to_id:
            return []
        
        user_id = self.user_name_to_id[user_name]
        return self.reference_signatures.get(user_id, [])
    
    def get_all_users(self):
        """
        Get list of all users
        
        Returns:
            List of user names
        """
        return list(self.user_name_to_id.keys())
    
    def remove_user(self, user_name):
        """
        Remove a user and all their reference signatures
        
        Args:
            user_name: Name of the user
            
        Returns:
            True if user was removed, False otherwise
        """
        if user_name not in self.user_name_to_id:
            return False
        
        user_id = self.user_name_to_id[user_name]
        
        # Remove the user's reference signatures
        if user_id in self.reference_signatures:
            del self.reference_signatures[user_id]
        
        # Remove the user name mapping
        del self.user_name_to_id[user_name]
        
        return True
    
    def clear_all_references(self):
        """
        Clear all reference signatures and users
        
        Returns:
            None
        """
        self.reference_signatures = {}
        self.user_name_to_id = {}
    
    def get_training_data(self, test_size=0.2):
        """
        Get training data for model training.
        Combines all user signatures and assigns labels.
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, y_train), where X_train is a list of feature vectors
            and y_train is a list of corresponding user IDs
        """
        X = []  # Feature vectors
        y = []  # Labels (user IDs)
        
        # Collect all reference signatures
        for user_id, signatures in self.reference_signatures.items():
            for signature in signatures:
                X.append(signature)
                y.append(user_id)
        
        # If there's not enough data, return everything for training
        if len(X) < 5:
            return X, y
        
        # Split data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        return X_train, y_train
    
    def generate_negative_samples(self):
        """
        Generate negative samples for training.
        Creates pairs of signatures from different users as negative examples.
        
        Returns:
            Tuple of (X_neg, y_neg), where X_neg is a list of feature vector pairs
            and y_neg is a list of 0s (indicating negative samples)
        """
        X_neg = []
        y_neg = []
        
        user_ids = list(self.reference_signatures.keys())
        
        # Need at least 2 users to generate negative samples
        if len(user_ids) < 2:
            return X_neg, y_neg
        
        # Generate negative sample pairs (signatures from different users)
        for i in range(len(user_ids)):
            for j in range(i+1, len(user_ids)):
                user1 = user_ids[i]
                user2 = user_ids[j]
                
                # Get a few random signatures from each user
                signatures1 = self.reference_signatures[user1]
                signatures2 = self.reference_signatures[user2]
                
                # Limit the number of negative samples
                max_samples = min(len(signatures1), len(signatures2), 5)
                
                for k in range(max_samples):
                    idx1 = random.randint(0, len(signatures1) - 1)
                    idx2 = random.randint(0, len(signatures2) - 1)
                    
                    # Create a negative sample (signatures from different users)
                    X_neg.append((signatures1[idx1], signatures2[idx2]))
                    y_neg.append(0)  # 0 indicates different users
        
        return X_neg, y_neg
