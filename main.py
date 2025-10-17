from fastapi import FastAPI
import pickle
import numpy as np

# Create the FastAPI app
app = FastAPI()

# --- Load your models once when the server starts ---
parent_model = pickle.load(open("parent_model.pkl", "rb"))
student_model = pickle.load(open("student_model.pkl", "rb"))
teacher_model = pickle.load(open("teacher_model.pkl", "rb"))

# --- Just a test route to check if backend runs ---
@app.get("/")
def root():
    return {"message": "Happiness Index Backend is running!"}

# --- Parent prediction endpoint ---
@app.post("/predict/parent")
def predict_parent(data: dict):
    x = np.array([[data["Unnamed0"], data["Quality"], data["Teacher_Quality"],
                   data["Books"], data["Classroom"], data["Food"],
                   data["Homework"], data["Understanding_subs"],
                   data["Meetings"], data["Continue_school"]]], dtype=float)
    pred = parent_model.predict(x)[0]
    return {"score": int(pred), "category": "Satisfied" if pred > 5 else "Unsatisfied"}

# --- Student prediction endpoint (adjust later) ---
@app.post("/predict/student")
def predict_student(data: dict):
    x = np.array([[data["feature1"], data["feature2"], data["feature3"]]], dtype=float)
    pred = student_model.predict(x)[0]
    return {"score": int(pred), "category": "Happy" if pred > 5 else "Stressed"}

# --- Teacher prediction endpoint (adjust later) ---
@app.post("/predict/teacher")
def predict_teacher(data: dict):
    x = np.array([[data["feature1"], data["feature2"], data["feature3"]]], dtype=float)
    pred = teacher_model.predict(x)[0]
    return {"score": int(pred), "category": "Satisfied" if pred > 5 else "Unsatisfied"}
