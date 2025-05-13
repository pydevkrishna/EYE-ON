import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os
from ultralytics import YOLO

# Define class names for safety equipment - matching the exact names from yolo_params.yaml
CLASS_NAMES = {
    0: "FireExtinguisher",
    1: "ToolBox",
    2: "OxygenTank"
}

def load_model():
    """Load the YOLO model with error handling"""
    try:
        # Use the trained model
        model_path = Path("runs/detect/train/weights/best.pt")
        if not model_path.exists():
            st.error("Trained model not found. Please ensure the model is trained first.")
            return None
            
        st.info(f"Loading model from: {model_path.absolute()}")
        model = YOLO(str(model_path))
        
        # Verify model loaded successfully
        if model is None:
            raise Exception("Model failed to load")
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process the image and return detections"""
    try:
        # Perform prediction with lower confidence threshold
        results = model.predict(
            image,
            conf=0.25,  # Lower confidence threshold
            iou=0.45,   # IOU threshold
            max_det=10  # Maximum detections
        )
        return results[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    st.title("Safety Equipment Detection")
    st.write("Upload an image to detect fire extinguishers, tool boxes, and oxygen tanks")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert the file to an image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Safety Equipment"):
                with st.spinner("Loading model and processing..."):
                    # Load model
                    model = load_model()
                    if model is None:
                        return
                        
                    # Process image
                    result = process_image(image, model)
                    if result is None:
                        return
                        
                    # Get the image with predictions drawn
                    img_with_boxes = result.plot()
                    
                    # Display the result
                    st.image(img_with_boxes, caption="Detection Results", use_column_width=True)
                    
                    # Display detection information
                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        class_name = CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
                        detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "box": box.xyxy[0].tolist()  # Get bounding box coordinates
                        })
                    
                    if detections:
                        st.write("Detected Safety Equipment:")
                        for det in detections:
                            st.write(f"- {det['class']} (Confidence: {det['confidence']:.2%})")
                            
                        # Add a summary section
                        st.write("---")
                        st.write("Detection Summary:")
                        st.write(f"Total safety equipment detected: {len(detections)}")
                        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                        st.write(f"Average confidence: {avg_confidence:.2%}")
                        
                        # Count by class
                        class_counts = {}
                        for det in detections:
                            class_name = det['class']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        st.write("\nBreakdown by type:")
                        for class_name, count in class_counts.items():
                            st.write(f"- {class_name}: {count}")
                            
                        # Show raw detection info for debugging
                        with st.expander("Debug Information"):
                            st.write("Raw detection boxes:")
                            for i, box in enumerate(result.boxes):
                                st.write(f"Box {i}:")
                                st.write(f"- Class ID: {int(box.cls)}")
                                st.write(f"- Confidence: {float(box.conf):.2%}")
                                st.write(f"- Bounding Box: {box.xyxy[0].tolist()}")
                                
                            # Show model information
                            st.write("\nModel Information:")
                            st.write(f"- Number of classes: {model.model.nc}")
                            st.write(f"- Model path: {model.model.pt_path}")
                            
                            # Show class mapping
                            st.write("\nClass Mapping:")
                            for cls_id, cls_name in CLASS_NAMES.items():
                                st.write(f"- Class {cls_id}: {cls_name}")
                    else:
                        st.write("No safety equipment detected")
                        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

if __name__ == "__main__":
    main() 