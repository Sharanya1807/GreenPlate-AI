# GreenPlate AI — Food Waste Prediction

## About the Project

GreenPlate AI is a machine learning-based application designed to predict food waste in kitchens and food service environments. It uses operational information such as meals served, kitchen staff, temperature, humidity, previous food waste, staff experience, and food category to estimate food waste in kilograms.

The goal is to help kitchen managers understand potential food waste and make more informed decisions about food preparation and waste reduction.

## Features

* **Food Waste Prediction:** Estimates food waste using a trained Random Forest Regression model.
* **Interactive Web Application:** Provides a user-friendly interface for entering kitchen data and viewing predictions.
* **Analytics Dashboard:** Displays food waste statistics and visual summaries.
* **Prediction History:** Keeps a history of predictions in the browser.
* **REST API:** Uses FastAPI to receive input data and return model predictions.
* **Cloud Deployment:** The prediction API is deployed using Render.

## Technology Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, Random Forest Regression
* **Backend:** FastAPI, Uvicorn
* **Data Processing:** Pandas
* **Model Serialization:** Joblib
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Render and Hatchable

## Input Features

The prediction model uses the following information:

* Number of meals served
* Number of kitchen staff
* Temperature in Celsius
* Humidity percentage
* Day of the week
* Special event indicator
* Previous food waste in kilograms
* Staff experience level
* Food waste category

## Model Evaluation

The model was previously evaluated with the following reported metrics:

* **Mean Absolute Error (MAE):** Approximately 4.79 kg
* **Root Mean Squared Error (RMSE):** Approximately 6.95 kg
* **R² Score:** Approximately 0.906

These are evaluation results, not a guarantee of accuracy for every future prediction. Performance depends on the dataset, evaluation method, and how closely new inputs resemble the training data.

## Live Application

* **Web Application:** https://greenplate-ai.hatchable.site
* **Prediction API:** https://greenplate-ai.onrender.com
* **API Documentation:** https://greenplate-ai.onrender.com/docs

## Running the API Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Sharanya1807/GreenPlate-AI.git
   cd GreenPlate-AI
   ```

2. Create and activate a virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:

   ```bash
   uvicorn api:app --reload
   ```

5. Open the interactive API documentation:

   http://127.0.0.1:8000/docs

Ensure that the trained model file and all required dependencies are available before running the API.

## Limitations

* Predictions are estimates and may differ from actual measured food waste.
* Model performance depends on the quality and representativeness of the dataset.
* Prediction history is stored in the browser and is not automatically shared across devices.
* The application supports decision-making but does not replace actual food-waste measurement.

## Future Improvements

* Evaluate predictions against real kitchen measurements.
* Collect additional data from different kitchens and food service environments.
* Improve input validation and model monitoring.
* Add downloadable reports and more detailed analytics.
* Explore recommendations for reducing food waste.

## Author

Developed as a machine learning and web application project focused on food waste prediction and reduction.
