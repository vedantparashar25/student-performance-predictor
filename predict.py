import pickle

model = pickle.load(open("model.pkl", "rb"))

hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
marks = float(input("Enter previous marks: "))

result = model.predict([[hours, attendance, marks]])

if result[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")