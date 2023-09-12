# Object-Detection-with-Weighted-Boxes-Fusion
ASSESMENT
## 1. Ensemble Model Creation:
Pretrained Models: Download pretrained YOLO and Faster R-CNN models from reputable sources or model zoos. Ensure they are compatible with the framework you plan to use (e.g., PyTorch or TensorFlow).
Model Ensemble: Develop a Python script to create an ensemble model. Use the ensemble_boxes library in Python to fuse the results from YOLO and Faster R-CNN. You can use the weighted_boxes_fusion function to combine bounding box predictions. This step does not require training; it's about loading and combining existing models.
Model Inference: Implement code to load both pretrained models, perform object detection with YOLO and Faster R-CNN, and then fuse the bounding box predictions using ensemble_boxes.
## 2. Docker Image Packaging:
Dockerfile: Create a Dockerfile for your ensemble model. Specify the necessary dependencies, including Python, PyTorch or TensorFlow (depending on the models), ensemble_boxes library, and any other required libraries.
Model Weights: Include the weights of the pretrained YOLO and Faster R-CNN models within the Docker image.
Code: Copy your ensemble model script and any associated configuration files into the Docker image.
Build and Push: Build the Docker image using the Dockerfile, and then push it to a container registry such as Docker Hub or Amazon Elastic Container Registry (ECR).
## 3. Web Interface:
Web Application: Develop a web application or API using a framework like Flask or FastAPI. This application will expose your ensemble model as a callable service.
API Endpoints: Define API endpoints that accept image input, perform object detection with the ensemble model, and return the detection results.
Model Loading: Load the ensemble model within your web application and ensure it's available for inference.
## 4. Generating Masks:
Inside Bounding Boxes: Implement code to generate masks within the bounding boxes for relevant objects. This step should use the Faster R-CNN predictions as input and create masks without relying on additional models like Mask R-CNN.
## 5. Deployment on Amazon SageMaker:
SageMaker Container: Create a custom SageMaker container based on your Docker image. This container should include your web application, the ensemble model, and any necessary dependencies.
SageMaker Deployment: Deploy your SageMaker endpoint using the custom container. SageMaker provides tools for model deployment, scaling, and monitoring.
Endpoint URL: Once deployed, your SageMaker endpoint will have a unique URL that can be used to make predictions.
## 6. GitHub Repository:
Create a GitHub repository to host your code, Dockerfiles, model weights, and any documentation.
Include detailed comments, explanations, and assumptions in your code and documentation to make the project understandable to others.
01. Clone this repository 
02. To Run the simple web interface built through streamlit on your local server run the following command on your terminal
  
  streamlit run app.py
  
 Notes:
 
Results are not promising yet for the weighted boxes fusion but i believe with some parameter tuning and using this ensembling method for a specific task oriented detection the results could be improved
