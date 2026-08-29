# 👔 Employee Attrition Classification

An end-to-end Machine Learning web application that predicts employee attrition risk based on workplace and demographic attributes. Built using **Scikit-Learn** and deployed with an interactive **Streamlit** user interface.

---

## 📌 Project Overview
Employee turnover can be costly for organizations. This project uses machine learning to classify whether an employee is at high risk of leaving a company. The web interface allows HR teams and managers to input employee data interactively and receive real-time attrition risk assessments and probabilities.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Machine Learning:** Scikit-Learn, Joblib
* **Data Manipulation:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## 📁 Repository Structure
```text
├── app.py                     # Streamlit application interface
├── attrition_model.pkl        # Trained Machine Learning model
├── scaler.pkl                 # StandardScaler object for feature inputs
├── employee_attrition_350.csv # Dataset used for training
├── employee_attrition_.ipynb  # Data exploration and training notebook
└── requirements.txt           # Python dependencies
---
## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
git clone https://github.com/swapnasaina274/employee-attrition-classification.git
cd employee-attrition-classification
```
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   4. Open your browser and navigate to `http://localhost:8501`.
      
