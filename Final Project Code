import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Creating Mock Datasets
np.random.seed(42)  # For reproducibility

# Dataset 1: Lunch Preferences
students = range(1, 101)
degrees = np.random.choice(['Business', 'Law', 'Engineering', 'Design'], 100)
genders = np.random.choice(['Male', 'Female'], 100)
lunch_spots = np.random.choice(['Honest Greens', 'Makkila', 'Pancake House', 'Five Guys', 'Starbucks', 'IE Cafeteria'], 100)

lunch_preferences_df = pd.DataFrame({
    'Student_ID': students,
    'Degree': degrees,
    'Gender': genders,
    'Lunch_Spot': lunch_spots
})

# Dataset 2: Dish Preferences
dishes = np.random.choice(['Burgers', 'Vegan Bowls', 'Pizzas', 'Salads', 'Pasta'], 100)
dish_preferences_df = pd.DataFrame({
    'Student_ID': students,
    'Preferred_Dish': dishes
})

# Part 2: Data Analysis and Visualization
# Lunch Spot Preferences by Degree
lunch_pref_degree = lunch_preferences_df.groupby(['Degree', 'Lunch_Spot']).size().unstack().fillna(0)
lunch_pref_degree.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Lunch Spot Preferences by Degree')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.legend(title='Lunch Spot')
plt.show()

# Lunch Spot Preferences by Gender
lunch_pref_gender = lunch_preferences_df.groupby(['Gender', 'Lunch_Spot']).size().unstack().fillna(0)
lunch_pref_gender.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Lunch Spot Preferences by Gender')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.legend(title='Lunch Spot')
plt.show()

# Dish Preferences Visualization
dish_pref_count = dish_preferences_df['Preferred_Dish'].value_counts()
dish_pref_count.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title('Preferred Dishes at the Cafeteria')
plt.ylabel('Votes')
plt.xticks(rotation=45)
plt.show()

# Part 3: Business Analysis (Example)
# Identifying the Most and Least Popular Lunch Spots
most_popular_spot = lunch_preferences_df['Lunch_Spot'].value_counts().idxmax()
least_popular_spot = lunch_preferences_df['Lunch_Spot'].value_counts().idxmin()

print(f"Most Popular Lunch Spot: {most_popular_spot}")
print(f"Least Popular Lunch Spot: {least_popular_spot}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Data Preparation
# Encoding categorical variables
encoder = LabelEncoder()
lunch_preferences_df['Degree_Encoded'] = encoder.fit_transform(lunch_preferences_df['Degree'])
lunch_preferences_df['Gender_Encoded'] = encoder.fit_transform(lunch_preferences_df['Gender'])
lunch_preferences_df['Lunch_Spot_Encoded'] = encoder.fit_transform(lunch_preferences_df['Lunch_Spot'])

# Preparing features and target
X = lunch_preferences_df[['Degree_Encoded', 'Gender_Encoded']]  # Features
y = lunch_preferences_df['Lunch_Spot_Encoded']  # Target

# Step 2: Feature Selection is done above

# Step 3: Model Training
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predicting the test set results
y_pred = model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

