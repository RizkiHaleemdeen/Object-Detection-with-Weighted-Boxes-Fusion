# Object-Detection-with-Weighted-Boxes-Fusion
## ASSESMENT
### 1. Ensemble Model Creation:
Pretrained Models: Downloaded pretrained YOLO and Faster R-CNN models from  model zoo.
Model Ensemble: Developed a Python script to create an ensemble model. Used the ensemble_boxes library in Python to fuse the results from YOLO and Faster R-CNN.
Model Inference: Implemented a code to load both pretrained models, perform object detection with YOLO and Faster R-CNN, and then fuse the bounding box predictions using ensemble_boxes.
### 2. Docker Image Packaging:
Dockerfile: Created a Dockerfile for the ensemble model. Specified the necessary dependencies
Code: Copied the ensemble model script and any associated configuration files into the Docker image.
Build and Push: Built the Docker image using the Dockerfile.
### 3. Web Interface:
Web Application: Developed a web application using Streamlit.
Model Loading: Loaded the ensemble model within the web application and made it available for inference
### 4. Generating Masks:
Inside Bounding Boxes: Implemented code to generate masks within the bounding boxes for predicted objects by using Faster R-CNN
### 5. GitHub Repository:
Created the GitHub repository to host your code, Dockerfiles, model weights, and any documentation.

## How to Run Using terminal
01. Clone this repository 
02. To Run the simple web interface built through streamlit on your local server run the following command on your terminal

  ``` pip install -r requirements.txt
      streamlit run app.py ```
  
 Notes:
 
Results are not promising yet for the weighted boxes fusion but i believe with some parameter tuning and using this ensembling method for a specific task oriented detection the results could be improved
