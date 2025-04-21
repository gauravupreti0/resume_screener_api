## AI-Powered Resume Screener

### Project introduction

A FastAPI-based API that accepts resumes (PDF or text), extracts features, classifies them (e.g., software engineer, data analyst, rejected), and returns a score or decision.

### Project Summary

This is a machine learning–based microservice that allows users to upload a resume file (PDF) and receive a prediction of the candidate's job role or domain — such as Software Engineer, Data Analyst, or Accountant.
It uses Natural Language Processing (NLP) and a trained ML classifier to analyze the content of the resume and make accurate, data-driven decisions.

### Key Features

- Accepts resumes in PDF format via API
- Extracts raw text using PDF parsers
- Uses TF-IDF + ML classifier to predict job role
- Built with FastAPI and Scikit-learn
- Supports local and containerized (Docker) deployment
- Designed and can be used for integration into HR tech stacks, ATS tools, and recruitment workflows

### What the Project Does

- Resume Upload
  A user (HR personnel, recruiter, or a system) uploads a resume file via the API endpoint.
- Text Extraction
  The backend reads the uploaded PDF and extracts all text using a Python PDF parser like pdfminer.six.
- Text Vectorization
  The raw text is transformed into machine-readable features using TF-IDF Vectorization, converting words into weighted numerical values based on their importance.
- ML Model Prediction
  A pre-trained Random Forest Classifier predicts the job category (e.g., "Software Engineer", "Data Analyst") based on the resume content.
- Response Delivery
  The predicted label is returned as a JSON response:
  ```
  {
  "prediction": "Software Engineer"
  }
  ```

### Tech Stack

Python, Fast API, Scikit-learn, Uvicorn
