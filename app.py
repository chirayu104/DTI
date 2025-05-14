import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
import uuid
import time
from datetime import datetime

# Import custom modules
from utils.image_processing import preprocess_image, normalize_image
from utils.feature_extraction import extract_features
from models.signature_model import SignatureModel
from utils.dataset_manager import DatasetManager

# Page configuration
st.set_page_config(
    page_title="Signature Verification System",
    page_icon="✍️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = SignatureModel()

if 'dataset_manager' not in st.session_state:
    st.session_state.dataset_manager = DatasetManager()

if 'reference_signatures' not in st.session_state:
    st.session_state.reference_signatures = {}

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Streamlit UI
st.title("Signature Verification System")

# Main navigation
tabs = st.tabs(["Verify Signature", "Manage Signatures", "Model Training"])

# Verify Signature Tab
with tabs[0]:
    st.header("Verify a Signature")
    
    if not st.session_state.model_trained:
        st.warning("Please add reference signatures and train the model first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload a Signature to Verify")
            uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "jpeg", "png"], key="verify_uploader")
            
            if uploaded_file is not None:
                # Create a temporary file to save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Read and preprocess the image
                try:
                    img = cv2.imread(temp_path)
                    if img is None:
                        st.error("Failed to load the image. Please try another file.")
                    else:
                        # Display the original uploaded image
                        st.image(uploaded_file, caption="Uploaded Signature", use_column_width=True)
                        
                        # Preprocess the image
                        processed_img = preprocess_image(img)
                        normalized_img = normalize_image(processed_img)
                        
                        # Extract features
                        features = extract_features(normalized_img)
                        
                        # Perform verification
                        verify_button = st.button("Verify Signature")
                        if verify_button:
                            with st.spinner("Verifying signature..."):
                                # Get the verification results
                                result, confidence = st.session_state.model.verify(features)
                                
                                # Display the result
                                if result:
                                    st.success(f"✅ Signature verified with {confidence:.2f}% confidence")
                                else:
                                    st.error(f"❌ Signature verification failed. Confidence: {confidence:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        with col2:
            st.subheader("Verification Process")
            st.info("""
            ### How it works
            1. The uploaded signature is preprocessed to enhance quality
            2. Key features are extracted from the signature
            3. These features are compared to reference signatures in the database
            4. The model calculates a similarity score
            5. If the score is above the threshold, the signature is verified
            
            ### What affects verification
            - Image quality
            - Signature consistency
            - Number of reference signatures
            - Signature complexity
            """)

# Manage Signatures Tab
with tabs[1]:
    st.header("Manage Reference Signatures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Reference Signature")
        user_name = st.text_input("User Name", key="user_name_input")
        
        uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "jpeg", "png"], key="reference_uploader")
        
        if uploaded_file is not None and user_name:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Reference Signature", use_column_width=True)
            
            add_button = st.button("Add as Reference Signature")
            
            if add_button:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Process and add the reference signature
                    img = cv2.imread(temp_path)
                    processed_img = preprocess_image(img)
                    normalized_img = normalize_image(processed_img)
                    features = extract_features(normalized_img)
                    
                    # Add to dataset manager
                    st.session_state.dataset_manager.add_reference_signature(user_name, features)
                    
                    # Update session state
                    if user_name not in st.session_state.reference_signatures:
                        st.session_state.reference_signatures[user_name] = []
                    st.session_state.reference_signatures[user_name].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    
                    st.success(f"Reference signature added for {user_name}")
                    
                    # Reset model trained flag since new data was added
                    st.session_state.model_trained = False
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    with col2:
        st.subheader("Current Reference Signatures")
        if not st.session_state.reference_signatures:
            st.info("No reference signatures added yet.")
        else:
            for user, timestamps in st.session_state.reference_signatures.items():
                with st.expander(f"{user} ({len(timestamps)} signatures)"):
                    for i, timestamp in enumerate(timestamps):
                        st.text(f"Signature {i+1}: Added on {timestamp}")
            
            if st.button("Clear All References"):
                st.session_state.reference_signatures = {}
                st.session_state.dataset_manager.clear_all_references()
                st.session_state.model_trained = False
                st.success("All reference signatures cleared")
                st.rerun()

# Model Training Tab
with tabs[2]:
    st.header("Train Verification Model")
    
    if not st.session_state.reference_signatures:
        st.warning("Please add reference signatures before training the model.")
    else:
        st.info(f"Found reference signatures for {len(st.session_state.reference_signatures)} users. " + 
                f"Total signatures: {sum(len(sigs) for sigs in st.session_state.reference_signatures.values())}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_button = st.button("Train Model")
            if train_button:
                with st.spinner("Training the model..."):
                    # Get training data from dataset manager
                    X_train, y_train = st.session_state.dataset_manager.get_training_data()
                    
                    if len(X_train) < 2:
                        st.error("Not enough reference signatures for training. Add at least 2 signatures.")
                    else:
                        # Train the model
                        training_progress = st.progress(0)
                        
                        # Mock training time for visualization purposes
                        for i in range(100):
                            time.sleep(0.05)
                            training_progress.progress(i + 1)
                        
                        # Actual training
                        st.session_state.model.train(X_train, y_train)
                        st.session_state.model_trained = True
                        st.success("Model trained successfully!")
        
        with col2:
            st.subheader("Model Information")
            if st.session_state.model_trained:
                st.success("Model is trained and ready for verification")
                st.info("""
                ### Model Details
                - Type: Siamese Neural Network
                - Feature Dimensions: 128
                - Verification Threshold: 0.7
                
                The model compares features of the input signature with stored references
                and calculates a similarity score to determine authenticity.
                """)
            else:
                st.warning("Model not yet trained")

# Footer
st.markdown("---")
st.caption("Signature Verification System | Powered by OpenCV and TensorFlow")
