# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create the dataset
data = {
    'Hours': [1, 2, 3, 4.5, 5, 6, 7, 8, 9, 10],
    'Scores': [20, 35, 50, 55, 60, 65, 75, 80, 85, 95]
}
df = pd.DataFrame(data)
print("Sample Data:\n", df)

# 3. Visualize the data
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hours', y='Scores', data=df, color='blue')
plt.title('Hours Studied vs Marks Scored')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.grid(True)
plt.show()

# 4. Split the data into training and testing sets
X = df[['Hours']]  # input
y = df['Scores']   # output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict using the model
y_pred = model.predict(X_test)

# 7. Compare actual vs predicted
result = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nActual vs Predicted:\n", result)

# 8. Plot the regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hours', y='Scores', data=df, label='Actual', color='blue')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Regression Line - Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.grid(True)
plt.show()

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 10. Predict for custom input (example: 7.5 hours)
custom_hours = [[7.5]]
predicted_score = model.predict(custom_hours)
print(f"\nPredicted score for 7.5 hours of study: {predicted_score[0]:.2f}")
