'''from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
from torchvision.transforms import transforms
from main import ConvNet  # Assuming your model is defined in a file named model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


app = Flask(__name__, static_folder='static')
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

df = pd.read_csv('dataset/training.csv')
tr = pd.read_csv('dataset/testing.csv')
treatments_df = pd.read_csv('treatments.csv')
treatments_dict = dict(zip(treatments_df['Disease'], treatments_df['Treatment']))

# List of symptoms
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Replace disease names with numbers in the training and testing datasets
df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
tr.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

# Training and testing data
X_train = df[l1]
y_train = df['prognosis']
X_test = tr[l1]
y_test = tr['prognosis']

# Load the trained model
model_path = 'model_xray_new.ckpt'
model = ConvNet(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transformer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('xray.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']

    # Check if the file is allowed
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not (image_file.filename.lower().endswith(tuple(allowed_extensions)) and image_file.content_type.startswith('image/')):
        return jsonify({'error': 'Invalid file format'})

    # Save the image to a temporary file
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Load the image, apply transformations, and make the prediction
    image = Image.open(temp_path).convert('RGB')
    image = transformer(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
    
    _, predicted_class = torch.max(output.data, 1)
    
    # Clean up the temporary file
    os.remove(temp_path)

    # Return the prediction result
    prediction = "Pneumonia" if predicted_class.item() == 1 else "Normal"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)'''
    
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision.transforms import transforms
from main import ConvNet  # Assuming your model is defined in a file named main.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder='static')

# Load the trained model
model_path = 'model_xray_new.ckpt'
model = ConvNet(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transformer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Load datasets
df = pd.read_csv('dataset/training.csv')
tr = pd.read_csv('dataset/testing.csv')
treatments_df = pd.read_csv('treatments.csv')
treatments_dict = dict(zip(treatments_df['Disease'], treatments_df['Treatment']))


# List of symptoms
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Replace disease names with numbers in the training and testing datasets
df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
tr.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

# Training and testing data
X_train = df[l1]
y_train = df['prognosis']
X_test = tr[l1]
y_test = tr['prognosis']

@app.route('/')
def home():
    return render_template('index.html', symptoms=sorted(l1))

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    symptoms = [request.form[f'Symptom{i}'] for i in range(1, 6)]

    # Convert symptom input to feature vector
    input_data = np.zeros(len(l1))
    for i, symptom in enumerate(symptoms):
        if symptom in l1:
            input_data[l1.index(symptom)] = 1

    # Initialize models
    clf_dt = tree.DecisionTreeClassifier()
    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_nb = GaussianNB()
    clf_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Train models
    clf_dt = clf_dt.fit(X_train, y_train)
    clf_rf = clf_rf.fit(X_train, y_train)
    clf_nb = clf_nb.fit(X_train, y_train)
    clf_knn = clf_knn.fit(X_train, y_train)

    # Make predictions
    predictions = {
        'Decision Tree': clf_dt.predict([input_data]),
        'Random Forest': clf_rf.predict([input_data]),
        'Naive Bayes': clf_nb.predict([input_data]),
        'K Nearest Neighbors': clf_knn.predict([input_data]),
    }

    # Evaluate accuracy
    y_true = [y_test.iloc[0]]  # Use the true label from your test set
    accuracies = {model: accuracy_score(y_true, pred) for model, pred in predictions.items()}

    # Determine the most accurate model
    most_accurate_model = max(accuracies, key=accuracies.get)
    predicted_disease = disease[predictions[most_accurate_model][0]]

    # Display results
    result = f"Predicted Disease: <strong>{predicted_disease}</strong><br>" \
             f"Most Accurate Model: <strong>{most_accurate_model}</strong><br>" 
            #  f"Accuracy: <strong>{accuracies[most_accurate_model]}</strong>"
    
    treatment_recommendation = treatments_dict.get(predicted_disease, "Treatment not available.")

    return render_template('result.html', result=result, treatment=treatment_recommendation)

@app.route('/xray')
def xray():
    return render_template('xray.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']

    # Check if the file is allowed
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not (image_file.filename.lower().endswith(tuple(allowed_extensions)) and image_file.content_type.startswith('image/')):
        return jsonify({'error': 'Invalid file format'})

    # Save the image to a temporary file
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Load the image, apply transformations, and make the prediction
    image = Image.open(temp_path).convert('RGB')
    image = transformer(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
    
    _, predicted_class = torch.max(output.data, 1)
    
    # Clean up the temporary file
    os.remove(temp_path)

    # Return the prediction result
    prediction = "Pneumonia" if predicted_class.item() == 1 else "Normal"
    return jsonify({'prediction': prediction})

if __name__ == '_main_':
    app.run(debug=True)
