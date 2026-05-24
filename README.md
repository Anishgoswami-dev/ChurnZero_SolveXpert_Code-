# 🚀 Banking Customer Churn Prediction System

## 📌 Overview
This project is a Machine Learning-based solution designed to predict **customer churn in the banking sector**. It analyzes customer behavior, transaction history, and engagement patterns to identify customers who are likely to leave (churn) in the near future.

The goal is to help banks take **proactive actions** to improve customer retention and reduce revenue loss.

---

## 📊 Dataset
The dataset contains:
- **8,101 training records**
- **2,026 test records**
- **97 features across 8 categories**

### 🔍 Feature Categories:
- Customer Profile (age, gender, income, etc.)
- Relationship & Tenure
- Account & Transaction Behavior
- Product Holding
- Credit Card & Loan Behavior
- Digital Banking Engagement
- Service & Complaint History
- Marketing & Retention Data

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Model Types:** Logistic Regression, Random Forest, XGBoost (or any used)  
- **Tools:** Jupyter Notebook / VS Code  

---

## 🔥 Features
- 📊 Data Cleaning & Preprocessing  
- 🧠 Feature Engineering  
- 🤖 Machine Learning Model Training  
- 📈 Model Evaluation (PR-AUC, F1 Score)  
- 🔍 Churn Prediction with Probability  
- 💡 Business Insights & Recommendations  

---

## 📈 Model Evaluation
The model is evaluated using:
- **Primary Metric:** PR-AUC  
- **Secondary Metric:** F1 Score  
- **Business Cost Analysis:**
  - False Negative Cost: ₹40,000  
  - False Positive Cost: ₹500  

---

## 📁 Project Structure

📦 Churn-Prediction
┣ 📂 data
┃ ┣ 📜 ChurnZero_Dataset_v1.csv
┃ ┗ 📜 ChurnZero_Test_v1.csv
┣ 📂 notebooks
┃ ┗ 📜 churn_model.ipynb
┣ 📂 src
┃ ┗ 📜 model.py
┣ 📂 outputs
┃ ┗ 📜 predictions.csv
┣ 📜 README.md
┗ 📜 requirements.txt


---

## ▶️ How to Run

### 1️⃣ Clone the Repository

git clone [https://github.com/your-username/churn-prediction.git](https://github.com/Anishgoswami-dev/ChurnZero_SolveXpert_Code-.git)
cd churn-prediction
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Model
python model.py
📌 Output

The final output file contains:

churn_prediction (0 or 1)
churn_probability (0–1)
💡 Key Insights
Customers with low engagement are more likely to churn
High complaint rates increase churn probability
Declining account balance is a strong churn indicator
🎯 Business Impact
Helps banks reduce customer churn
Enables targeted marketing campaigns
Improves customer satisfaction & retention
👨‍💻 Author

Anish Goswami
📌 B.Tech CSE Aspirant | ML Enthusiast.
### Screenshot:- 
<img width="1900" height="890" alt="image" src="https://github.com/user-attachments/assets/5712cef9-7dcd-49a6-ba02-b12c44f9d98a" />

