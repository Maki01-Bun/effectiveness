import joblib
import pandas as pd

# Load model
model = joblib.load("random_forest_subsidy.pkl")

print("\n===== BENEFICIARY FEEDBACK FORM =====\n")

farmer_name = input("Farmer Name: ")
subsidy_type = input("Subsidy Type: ")

farm_size = float(input("Farm Size (ha): "))
crop_yield_before = float(input("Crop Yield Before: "))
crop_yield_after = float(input("Crop Yield After: "))

income_before = float(input("Income Before: "))
income_after = float(input("Income After: "))

pest = input("Pest (Yes/No): ")
calamity = input("Calamity (Yes/No): ")

print("\n===== SURVEY FORM =====\n")
print("1 = Strongly Disagree")
print("2 = Disagree")
print("3 = Neutral")
print("4 = Agree")
print("5 = Strongly Agree\n")

q1 = int(input("1. Improved farm productivity: "))
q2 = int(input("2. Reduced farming expenses: "))
q3 = int(input("3. Increased income: "))
q4 = int(input("4. Improved farming practices: "))
q5 = int(input("5. Satisfied with the subsidy program: "))
q6 = int(input("6. Recommend to other farmers: "))

feedback_score = (q1 + q2 + q3 + q4 + q5 + q6) / 6

print("\nFeedback Score:", round(feedback_score, 2))

data = pd.DataFrame([{
    "Subsidy Type": subsidy_type,
    "Farm Size (ha)": farm_size,
    "Crop Yield Before": crop_yield_before,
    "Crop Yield After": crop_yield_after,
    "Income Before": income_before,
    "Income After": income_after,
    "Feedback Score": feedback_score,
    "Pest": pest,
    "Calamity": calamity
}])

prediction = model.predict(data)[0]

print("\n============================")
print("Farmer:", farmer_name)
print("Predicted Effectiveness:", prediction)
print("============================")