from fastapi import FastAPI, UploadFile, File
from app.resume_parser import extract_text_from_resume
import joblib
import os

app = FastAPI()
model = joblib.load('model/resume_classifier.pkl')
vectorizer = joblib.load('model/resume_vectorizer.pkl')

@app.post('/api/')
async def classify_resume(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(await file.read())
        
    text = extract_text_from_resume(file_path)
    os.remove(file_path)
    
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    
    return {'prediction' : prediction[0]}