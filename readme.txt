A Python project completing the CodeOrbitTech Machine Learning internship tasks (1 Month tier) using the classic Iris dataset. It covers data preprocessing and EDA, a simple classification model, and a regression model for prediction.

Project Title

Iris Dataset Analysis: EDA, Classification and Regression

Tasks Covered
Task	Description	Output
1	Data Preprocessing & Exploratory Data Analysis	Histograms, correlation heatmap, pairplot, boxplots
2	Simple Classification Model	Logistic Regression and Decision Tree, with accuracy, precision, recall and confusion matrices
3	Regression Model for Prediction	Linear and polynomial regression predicting petal width from petal length
Technologies Used
Python 3.9+
pandas: data handling and cleaning
NumPy: numerical operations
Matplotlib: plotting
Seaborn: statistical visualizations
scikit-learn: dataset loading, models and evaluation metrics
Project Structure
.
├── iris_ml_tasks.py        # Main script (Tasks 1, 2 and 3)
├── README.md               # Project documentation
├── eda_histograms.png      # Generated: feature distributions
├── eda_correlation.png     # Generated: correlation heatmap
├── eda_pairplot.png        # Generated: pairplot
├── eda_boxplots.png        # Generated: boxplots
├── confusion_*.png         # Generated: confusion matrices
└── regression_results.png  # Generated: regression fits and predictions
Results Summary
Classification: Both Logistic Regression and Decision Tree reached about 96.7% accuracy. Setosa was classified perfectly; the few errors were between versicolor and virginica.
Regression: Petal length predicts petal width well (R² ≈ 0.925). The polynomial model gave no meaningful improvement over the linear one.
GitHub Repository Link
