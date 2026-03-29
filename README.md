# Mushroom-Classification-Web-App-using-Streamlit-and-Machine-Learning

## Abstract
This project focuses on building an interactive **Machine Learning web application** using **Streamlit** to classify mushrooms as **edible** or **poisonous** based on their physical and categorical characteristics.  

The application allows users to train and evaluate multiple machine learning classification models and compare their performance using standard evaluation metrics and visualization plots.

The project integrates:

- Data preprocessing using categorical encoding  
- Machine learning classification models  
- Interactive web deployment using Streamlit  
- Visual performance analysis using classification metrics  

---

## 1. Problem Statement
Identifying whether a mushroom is **edible** or **poisonous** is an important real-world classification problem because consuming poisonous mushrooms can cause severe health issues or even death.  

Since many mushroom species have similar appearances, manual identification can be difficult, especially for non-experts. Therefore, there is a need for an **automated intelligent classification system** that can predict whether a mushroom is safe to consume.

This project aims to:

1. Load and preprocess the mushroom dataset  
2. Train multiple machine learning classification models  
3. Evaluate models using multiple performance metrics  
4. Build an interactive web application for classification and visualization  

---
### **Application URL**
- **Local Deployment:** [http://localhost:8501/](http://localhost:8501/)

## 2. Simulation / Development Tool
**Web Framework:** `Streamlit`  
**Machine Learning Library:** `Scikit-learn`  
**Programming Language:** `Python`

### Why Streamlit?
- Easy conversion of Python scripts into web apps  
- Interactive user interface without advanced frontend coding  
- Fast visualization of machine learning results  
- Simple integration with Pandas, Matplotlib, and Scikit-learn  

### Additional Libraries Used
- `pandas` → Data loading and preprocessing  
- `matplotlib` → Graph plotting and visualization  
- `scikit-learn` → Model training, testing, and evaluation  

---

## 3. Methodology

### Step 1: Dataset Loading
The mushroom dataset is loaded from a CSV file (`mushrooms.csv`).

The dataset contains mushroom-related categorical attributes such as:

- Cap Shape  
- Cap Color  
- Odor  
- Gill Size  
- Habitat  
- Stalk Shape  
- And more...

The target variable represents whether a mushroom is:

- **Edible**
- **Poisonous**

---

### Step 2: Data Preprocessing
Since the mushroom dataset contains categorical values, preprocessing is required before model training.

#### Preprocessing Steps:
1. Load dataset using **Pandas**
2. Encode categorical columns using **Label Encoding**
3. Separate:
   - **Features (X)**
   - **Target Variable (y)**
4. Split dataset into:
   - **80% Training Data**
   - **20% Testing Data**

---

### Step 3: Machine Learning Models
The following classification models were trained and evaluated:

1. Support Vector Machine (**SVM**)  
2. Logistic Regression  
3. Random Forest Classifier  

These models were selected because they are commonly used and effective for binary classification tasks.

---

### Step 4: Evaluation Metrics
The models were evaluated using the following classification metrics:

| Metric | Type | Description |
|--------|------|-------------|
| Accuracy | Benefit | Overall correctness of predictions |
| Precision | Benefit | Correctness of positive predictions |
| Recall | Benefit | Ability to identify actual positive cases |

### Additional Visual Metrics:
- **Confusion Matrix**
- **ROC Curve**
- **Precision-Recall Curve**

These metrics help analyze model performance from multiple perspectives.

---

### Step 5: Streamlit Web Application
The trained models are integrated into a **Streamlit web app** where the user can:

1. Select a machine learning classifier  
2. Set model hyperparameters  
3. Train the model  
4. View evaluation metrics  
5. Visualize performance plots  
6. Display the encoded dataset  

---

## 4. Application Workflow

### Input
The user interacts with the Streamlit web interface and selects:

- Classifier type
- Hyperparameters
- Evaluation metrics to display

### Processing
The selected model is trained on the preprocessed mushroom dataset.

### Output
The application displays:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

## 5. Models Used

### 1. Support Vector Machine (SVM)
Support Vector Machine is a supervised learning algorithm used for binary classification tasks. It works by finding the optimal hyperplane that best separates the classes.

### 2. Logistic Regression
Logistic Regression is a simple and efficient classification algorithm commonly used for binary prediction problems.

### 3. Random Forest
Random Forest is an ensemble learning algorithm that uses multiple decision trees to improve classification accuracy and reduce overfitting.

---

## 6. Results

### Example Model Evaluation Output

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Support Vector Machine | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 0.95 | 0.96 | 0.94 |
| Random Forest | 1.00 | 1.00 | 1.00 |

> Note: The exact values may vary depending on the dataset split and hyperparameters selected.

### Observed Results
- All three models were successfully trained and tested
- Tree-based and kernel-based models showed excellent classification performance
- The Streamlit interface successfully displayed:
  - Numerical evaluation metrics
  - Performance graphs
  - Encoded dataset preview

---

## 7. Result Graphs

The application can display the following visual outputs:

- **Confusion Matrix**
- **ROC Curve**
- **Precision-Recall Curve**

These graphs help in understanding:

- Prediction accuracy
- False positives and false negatives
- Trade-off between precision and recall
- Classifier discrimination ability

> Interpretation: Models like **Random Forest** and **SVM** generally perform very well on the mushroom dataset because the dataset is highly structured and separable.

---

## 8. Discussion
- Machine learning can effectively classify mushrooms into edible and poisonous categories.
- Label encoding makes categorical data usable for machine learning models.
- SVM performs well for binary classification with clear class separation.
- Random Forest handles complex feature relationships efficiently.
- Logistic Regression provides a simple baseline model.
- Visualization metrics improve understanding of classifier behavior.

This project demonstrates how machine learning can be integrated into a user-friendly web application for practical classification tasks.

---

## 9. Conclusion
This project successfully demonstrates:

- Mushroom classification using machine learning  
- Data preprocessing using categorical encoding  
- Training and evaluation of multiple classification models  
- Deployment of a machine learning web application using Streamlit  

The project shows that machine learning algorithms such as:

- **Support Vector Machine**
- **Logistic Regression**
- **Random Forest**

can accurately classify mushrooms as **edible** or **poisonous**.

Overall, this project is a practical example of applying **Machine Learning + Data Preprocessing + Web Deployment** to solve a real-world classification problem.

---

## 10. Future Improvements
Possible future improvements for this project include:

- Adding more classification algorithms  
- Hyperparameter tuning for improved performance  
- Cross-validation for more robust evaluation  
- Feature importance analysis  
- Better UI/UX design in Streamlit  
- Cloud deployment using:
  - Streamlit Cloud
  - Render
  - Heroku  

The project can also be extended into an **image-based mushroom classification system** using **Deep Learning** and **Computer Vision**.

---
