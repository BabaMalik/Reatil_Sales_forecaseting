
# Retail Sales Forecasting

## 📌 Project Overview
Retail sales forecasting is essential for optimizing inventory, managing supply chains, and increasing revenue. This project predicts sales for thousands of product families sold in Favorita stores in Ecuador. Using machine learning techniques like **SGDRegressor**, **LSTM (Long Short-Term Memory Networks)**, and **SVM (Support Vector Machine)**, we aim to improve sales prediction accuracy.

## 🏗 Dataset Description
The dataset consists of multiple CSV files:
- **train.csv** → Historical sales data for different product families across stores.
- **test.csv** → Contains the same structure as train.csv but without sales values (used for prediction).
- **stores.csv** → Metadata about store locations and types.
- **oil.csv** → Daily oil price data (Ecuador’s economy depends on oil prices).
- **holidays_events.csv** → List of holidays and events that may impact sales.
- **transactions.csv** → Number of daily transactions for each store.

## 🛠 Technologies & Libraries Used
- **Python** (Data Analysis & Modeling)
- **Jupyter Notebook** (Code Execution)
- **Pandas, NumPy** (Data Manipulation)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-Learn** (Machine Learning - SGDRegressor, SVM)
- **TensorFlow, Keras** (Deep Learning - LSTM)
- **TQDM** (Progress tracking)

## 🚀 How to Run the Project
### **1️⃣ Install Dependencies**
Run the following command in your terminal:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tqdm
```
### **2️⃣ Clone the Repository & Navigate to the Folder**
```bash
git clone https://github.com/yourusername/retail-sales-forecasting.git
cd retail-sales-forecasting
```
### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook retail_sales_forecasting.ipynb
```

## 🔍 Exploratory Data Analysis (EDA)
We performed the following analyses to understand the dataset:
- **Sales Trends Over Time** 📈
- **Impact of Promotions on Sales** 🛒
- **Effect of Holidays on Revenue** 🎉
- **Oil Price vs. Sales Correlation** ⛽

## 🧠 Machine Learning Models
We implemented **three models** to predict sales:

### **1️⃣ SGDRegressor (Fast ML Model)**
- Applied **StandardScaler** to normalize features.
- Trained a **Stochastic Gradient Descent (SGD) Regressor**.
- Performance Metric: **Mean Squared Error (MSE)**

### **2️⃣ LSTM (Deep Learning Model for Time-Series)**
- Preprocessed data using **MinMaxScaler**.
- Reshaped input to **(samples, time-steps, features)** format.
- Built an **LSTM network with 2 layers**.
- Trained using **Adam optimizer**.

### **3️⃣ SVM (Support Vector Machine)**
- Standardized features using **StandardScaler**.
- Implemented **SVR (Support Vector Regression)** with **RBF kernel**.
- Performance Metric: **Mean Squared Error (MSE)**.
- Note: SVM was **not yet executed** in the notebook.

## 📊 Results & Comparison
| Model | Mean Squared Error (MSE) |
|--------|----------------------|
| **SGDRegressor** | 808,277.18 |
| **LSTM** | 11,396.67 |
| **SVM** | (Pending Execution) |

🔹 **LSTM performed significantly better than SGDRegressor** because it captures time-series dependencies more effectively.

## 📂 Project Structure
```
Retail_Sales_Forecasting/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   ├── transactions.csv
│
├── notebooks/
│   ├── retail_sales_forecasting.ipynb
│
├── models/
│   ├── sgd_model.pkl
│   ├── lstm_model.h5
│   ├── svm_model.pkl (Pending Execution)
│
├── images/
│   ├── sales_trend.png
│   ├── model_comparison.png
│
├── README.md
└── requirements.txt
```

## 🤝 Contributing & Future Work
### **Future Improvements:**
✅ **Try XGBoost or Random Forest models** for better accuracy.
✅ **Hyperparameter tuning** for LSTM to further optimize predictions.
✅ **Execute and analyze the performance of SVM.**
✅ **Deploy the model as a web app using Flask or FastAPI**.

### **Want to Contribute?**
Feel free to **fork the repository**, make improvements, and submit a **pull request**!

📧 Contact: [Your Email] | 💻 GitHub: [YourUsername]

---
🚀 **Thank you for checking out the project!** Hope it helps in understanding retail sales forecasting! 🔥


#### https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
