# Bank Customer Churn Prediction Project

This project focuses on predicting whether customers will churn from a bank using a a machine learning model, and it is built and trained using MLflow for tracking experiments. The project was implemented with keras, and the application was deployed using Streamlit.

## Project Structure

- **`app.py`**: The main script for running the Streamlit app.
- **`Dockerfile`**: Contains instructions to set up the project environment using Docker for deployment.
- **`Bank Customer Churn Prediction.csv`**: Dataset used for training the model.
- **`DEBI_PROJECT_MLflow_NN.ipynb`**: Jupyter notebook for experimenting with neural network models, tracked using MLflow.
- **`best_models`**: Folder containing saved models.
- **`mlflow_results`**: Folder storing MLflow experiment results.
- **`requirements.txt`**: Python dependencies required for the project.


## Problem Statement

Customer churn is a critical issue for banks as retaining customers is essential for long-term profitability. This project aims to predict if a customer will churn (leave the bank) based on various features, allowing the bank to take preemptive actions.


## Dataset

The dataset includes features like customer age, balance, credit score, and geographic information to predict churn. The target variable is binary, where:
- `1` indicates the customer has churned.
- `0` indicates the customer has not churned.

**Challenges with the Dataset:**
- The data was unbalanced, with more customers in the 'not churned' category than in the 'churned' category. This required implementing strategies like changing class weights and using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.


## Visualizations
We are currently developing **dashboards** to visualize insights from the data and model predictions. This will include:
- **Customer segmentation**
- **Feature importance**
- **Churn distribution across different demographics**

This section will be updated once the dashboard development is completed.

## Model Training

The model was built using a neural network with Keras and TensorFlow. The training process was tracked using MLflow, where multiple configurations were tested to find the best-performing model. Metrics like AUC (Area Under the Curve), accuracy, precision, and recall were used to evaluate model performance.

## Results
The best performing model had:
- **Validation Accuracy**: 80.00%
- **AUC**: 86.11%
- **Precision**: 49.00%
- **Recall**: 76.00%

  
 ![WhatsApp Image 2024-10-07 at 23 22 28_e77706a3](https://github.com/user-attachments/assets/ac055110-187c-4c96-aa10-1102e6531123)


## Deployment

The application was deployed using Streamlit, providing an interactive interface where users can input customer details and get predictions on whether a customer is likely to churn.
https://csj2ctxuwfypzkr4n9xhjc.streamlit.app/


## How to Run

### Locally
1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Using Docker
1. Build the Docker image:
   ```
   docker build -t bank_customer_churn_prediction . 
   ```
2. Run the Docker container:
   ```
   docker run -p 5000:5000  bank_customer_churn_prediction
   ```

## Issues Faced

- **Unbalanced Dataset**: The dataset had a significant imbalance, making it challenging to get accurate predictions for the minority class (churned customers). This was addressed by experimenting with different class weights and using SMOTE to generate synthetic samples for the minority class.
- **Model Fine-tuning**: Various configurations of class weights and model parameters were tested to optimize performance.
- **Deployment**: Ensuring that the app runs smoothly on Streamlit and Docker required careful configuration of dependencies.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Keras
  - TensorFlow
  - Streamlit
  - MLflow
  - Pandas, Numpy, Matplotlib
- **Tools**:
  - Docker
  - Git & GitHub for version control














