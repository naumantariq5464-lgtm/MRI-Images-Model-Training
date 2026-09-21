# MRI Brain Tumor Detection AI

An end-to-end AI project for detecting Pituitary Tumors from MRI scans, featuring a trained deep learning model, a FastAPI backend, and a React (Vite) frontend with Grad-CAM explainability.

## Project Structure

- **`models/`**: Contains the trained Keras model (`pituitary_model.keras`). MobileNetV2 was used for feature extraction and fine-tuned for this specific binary classification task.
- **`training/`**: Python scripts used to preprocess the dataset, train the model, evaluate performance, and generate Grad-CAM heatmaps.
- **`backend/`**: A FastAPI application that serves the AI model. It provides endpoints for health checks and image prediction/inference.
- **`frontend/`**: A React application built with Vite. It features a premium Black & White UI where users can upload MRI scans and view real-time predictions alongside Grad-CAM visualization heatmaps.

## Features

- **Binary Classification**: AI model trained to distinguish between Normal Brain MRIs and Pituitary Tumor MRIs.
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to show exactly which regions of the MRI the model focused on to make its decision.
- **Modern UI/UX**: Glassmorphism design, animated confidence meters, and drag-and-drop file uploading.

## Technologies Used

- **Deep Learning**: TensorFlow, Keras, MobileNetV2
- **Image Processing**: OpenCV (cv2)
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: React, Vite, CSS3
