import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

class SignatureModel:
    """
    A model for signature verification using feature vectors.
    This lightweight model uses machine learning to compare signatures.
    """
    
    def __init__(self):
        """Initialize the signature verification model"""
        # Initialize the classifier
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Initialize the KNN classifier as a backup for few samples
        self.knn = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=1))
        ])
        
        # Initialize the reference signatures storage
        self.reference_features = {}
        
        # Define the verification threshold
        self.threshold = 0.7
        
        # Flag to track if the model is trained
        self.is_trained = False
        
        # Track the training model type (classifier or distance-based)
        self.model_type = None
        
    def train(self, X, y):
        """
        Train the model using the provided feature vectors and labels.
        
        Args:
            X: List of feature vectors from reference signatures
            y: List of labels (user IDs)
            
        Returns:
            None
        """
        X = np.array(X)
        y = np.array(y)
        
        # Determine which model type to use based on sample count
        unique_users = np.unique(y)
        
        if len(X) >= 10 and len(unique_users) >= 2:
            # Use classifier approach for multiple users with sufficient samples
            self.classifier.fit(X, y)
            self.model_type = 'classifier'
        else:
            # Use KNN for few samples
            self.knn.fit(X, y)
            self.model_type = 'knn'
        
        # Store reference feature vectors for each user
        self.reference_features = {}
        for i, user_id in enumerate(y):
            if user_id not in self.reference_features:
                self.reference_features[user_id] = []
            self.reference_features[user_id].append(X[i])
        
        # Convert lists to numpy arrays for faster computation
        for user_id in self.reference_features:
            self.reference_features[user_id] = np.array(self.reference_features[user_id])
        
        self.is_trained = True
    
    def verify(self, feature_vector):
        """
        Verify a signature against the reference signatures.
        
        Args:
            feature_vector: Feature vector of the signature to verify
            
        Returns:
            Tuple of (verification result, confidence score)
        """
        if not self.is_trained:
            return False, 0.0
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        if self.model_type == 'classifier':
            # Use the trained classifier for prediction
            user_probabilities = self.classifier.predict_proba(feature_vector)[0]
            max_confidence = np.max(user_probabilities) * 100
            
            # If confidence is above threshold, consider it verified
            if max_confidence / 100 >= self.threshold:
                return True, max_confidence
            else:
                return False, max_confidence
        else:
            # Use distance-based approach (for few samples)
            # Find the closest reference signature using cosine similarity
            max_similarity = 0
            max_user = None
            
            for user_id, references in self.reference_features.items():
                # Calculate average similarity with all user's references
                similarities = []
                for ref in references:
                    # Ensure shape compatibility
                    ref_reshaped = ref.reshape(1, -1)
                    similarity = cosine_similarity(feature_vector, ref_reshaped)[0][0]
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    max_user = user_id
            
            # Convert similarity to confidence percentage
            confidence = max_similarity * 100
            
            # Compare against threshold
            if max_similarity >= self.threshold:
                return True, confidence
            else:
                return False, confidence
