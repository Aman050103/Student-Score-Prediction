import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("📘 Student Score Predictor")
st.subheader("🔍 Predict your exam score based on study hours!")

st.markdown(
    """
    This simple app uses **Linear Regression** to predict a student's marks based on the number of hours studied.  
    Model is trained on a small sample dataset.
    """
)

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4.5, 5, 6, 7, 8, 9, 10],
    'Scores': [20, 35, 50, 55, 60, 65, 75, 80, 85, 95]
}
df = pd.DataFrame(data)

# Show sample data
with st.expander("📊 View Sample Data"):
    st.dataframe(df)

# Train model
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction input
hours_input = st.slider("📚 How many hours did you study?", 0.0, 12.0, 1.0, step=0.5,
                        help="Use the slider to select how many hours you studied.")

if st.button("🎯 Predict My Score"):
    predicted = model.predict([[hours_input]])
    final_score = min(max(predicted[0], 0), 100)  # Keep prediction between 0 and 100
    st.success(f"✅ Predicted Score: **{final_score:.2f} marks** for {hours_input} hours of study.")

    if predicted[0] > 100:
        st.warning("⚠️ Note: Raw prediction exceeds 100. Score capped at 100.")

# Regression plot
with st.expander("📈 Show Regression Plot"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Hours', y='Scores', data=df, color='blue', label='Actual Data')
    plt.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks Scored')
    plt.title('Regression Analysis: Study Hours vs Marks')
    plt.legend()
    st.pyplot(plt)

# Footer
st.markdown("""---""")
st.caption("🔧 Created with ❤️ using Streamlit & scikit-learn | Aman Singh")
