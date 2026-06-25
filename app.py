from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from preprocess import preprocess_incoming_data
from predict import get_predictions
import uvicorn
import os

app = FastAPI(title="Customer Churn Prediction API")

# Ensure templates and static dirs exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class CustomerData(BaseModel):
    customerID: str = "NewCustomer"
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 0.0
    TotalCharges: float = 0.0

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_api")
async def predict_api(customer: CustomerData):
    customer_dict = customer.model_dump()
    try:
        processed_data = preprocess_incoming_data(customer_dict)
        predictions = get_predictions(processed_data)
        
        risk_level = "High" if predictions['churn_probability'] > 0.5 else "Low"
        if predictions['churn_probability'] > 0.7:
            risk_level = "Critical"
            
        return {
            "success": True,
            "data": {
                **predictions,
                "risk_level": risk_level
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
