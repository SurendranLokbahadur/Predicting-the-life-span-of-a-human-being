from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to train the regression model
def train_regression_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize a random forest regressor model
    model = RandomForestRegressor(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error (MSE) as a metric
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model

# Function to predict life span based on user inputs
def predict_life_span(model, input_data):
    # Reshape input_data into a numpy array of shape (1, 10)
    X = np.array(input_data).reshape(1, -1)
    
    # Predict life span using the trained model
    predicted_life_span = model.predict(X)[0]
    
    return predicted_life_span

# Sample synthetic dataset (replace with real data)
# Here, let's use more realistic ranges and relationships between factors and life span
np.random.seed(42)
X = np.random.randint(1, 9, size=(100, 10))  # Example features (replace with real dataset)
y = np.random.rand(100) * 80 + 20  # Example target (replace with real dataset)

# Train the regression model on the synthetic data
model = train_regression_model(X, y)

# Function to get user input with options
def get_user_input_with_options(prompt, options):
    while True:
        print(prompt)
        for idx, option in enumerate(options, start=1):
            print(f"{idx}. {option}")
        choice = input(f"Enter your choice (1-{len(options)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        else:
            print("Invalid input. Please enter a number corresponding to the options.")

# Provide options and explanations for each input factor
options = {
    'Genetics': ["Very poor", "Poor", "Below average", "Average", "Above average", "Good", "Very good", "Excellent"],
    'Exercise': ["None", "Rarely", "Occasionally", "Moderately", "Regularly", "Frequently", "Daily", "Intensively"],
    'The Environment': ["Polluted and unsafe", "Unsafe", "Average", "Moderately safe", "Safe", "Very safe", "Ideal", "Optimal"],
    'Lifestyle': ["Extremely unhealthy", "Unhealthy", "Below average", "Average", "Above average", "Healthy", "Very healthy", "Optimal"],
    'Access to Health Care': ["No access", "Very limited", "Limited", "Basic", "Moderate", "Good", "Excellent", "Exceptional"],
    'Diet and Nutrition': ["Very poor", "Poor", "Below average", "Average", "Above average", "Good", "Very good", "Excellent"],
    'Hygiene': ["Very poor", "Poor", "Below average", "Average", "Above average", "Good", "Very good", "Excellent"],
    'Socio-economic Status': ["Very low", "Low", "Below average", "Average", "Above average", "Good", "Very good", "Excellent"],
    'Education Level': ["None", "Primary school", "Secondary school", "High school diploma", "Associate degree", "Bachelor's degree", "Master's degree", "Ph.D."],
    'Medical History': ["Very poor", "Poor", "Below average", "Average", "Above average", "Good", "Very good", "Excellent"]
}

# Prompt user for inputs with options and explanations
genetics = get_user_input_with_options("Genetics:", options['Genetics'])
exercise = get_user_input_with_options("Exercise:", options['Exercise'])
environment = get_user_input_with_options("The Environment:", options['The Environment'])
lifestyle = get_user_input_with_options("Lifestyle:", options['Lifestyle'])
healthcare_access = get_user_input_with_options("Access to Health Care:", options['Access to Health Care'])
diet_nutrition = get_user_input_with_options("Diet and Nutrition:", options['Diet and Nutrition'])
hygiene = get_user_input_with_options("Hygiene:", options['Hygiene'])
socioeconomic_status = get_user_input_with_options("Socio-economic Status:", options['Socio-economic Status'])
education_level = get_user_input_with_options("Education Level:", options['Education Level'])
medical_history = get_user_input_with_options("Medical History:", options['Medical History'])

# Predict life span using the trained model and user inputs
input_data = [genetics, exercise, environment, lifestyle, healthcare_access,
              diet_nutrition, hygiene, socioeconomic_status, education_level, medical_history]
predicted_life_span = predict_life_span(model, input_data)

print(f"\nPredicted Life Span: {predicted_life_span:.2f} years")
