import io
import matplotlib
matplotlib.use('Agg') 

from flask import Flask, request, render_template, jsonify, send_file    # Import jsonify
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import pandas as pd
import pickle
import google.generativeai as genai
from flask_cors import CORS
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()

# flask app
app = Flask(__name__)
CORS(app)

# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

def GenerateResponse(input_text):
    """Generates AI response and ensures proper formatting."""
    try:
        response = model.generate_content([
            "You are a healthcare model. Answer concisely and properly in a structured format.",
            f"User input: {input_text}",
            "Use bullet points (•) where necessary and preserve formatting."
        ])
        response_text = response.text.strip()

        # Clean up unwanted spaces, line breaks, and extra bullet points
        response_text = response_text.replace("\n\n", "\n")  # Remove extra blank lines
        response_text = response_text.replace("\n", "\n • ")  # Add bullet points for each line

        # Remove leading/trailing spaces from the response
        return response_text.strip()
    except Exception:
        return "Sorry, I am facing issues retrieving a response."
def ask_gemini(question):
    """Fetches a response from Gemini AI."""
    model = genai.GenerativeModel("gemini-pro")
    try:
        response = model.generate_content(question)
        return response.text if response else "No response from AI."
    except Exception as e:
        return f"Error: {str(e)}"

def save_chat_history(user_input, bot_response):
    """Save chat history to a file."""
    with open("chat_history.txt", "a", encoding="utf-8") as file:
        file.write(f"User: {user_input}\nBot: {bot_response}\n\n")

@app.route('/bot')
def bot():
    return render_template("bot.html")  # Serve frontend UI

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "I didn't understand that. Can you clarify?"})

    bot_response = GenerateResponse(user_input)
    
    # Save chat history
    save_chat_history(user_input, bot_response)

    return jsonify({"response": bot_response})

# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
severity_data = pd.read_csv('datasets/Symptom-severity.csv')

# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))


#============================================================
# custome and helping functions
#==========================helper funtions================
def get_gemini_details(disease):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Provide detailed information on {disease}, including symptoms, causes, and treatments.")
    return response.text if response else "No additional information available."

def helper(dis, severity_score):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values] if not pre.empty else []

    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    
    # Get additional details from Gemini
    gemini_details = get_gemini_details(dis)

    return desc, pre, med, die, wrkout, severity_score, gemini_details

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Calculate severity score for user symptoms
def calculate_severity(patient_symptoms):
    matched_symptoms = severity_data[severity_data['Symptom'].isin(patient_symptoms)]
    return matched_symptoms['weight'].sum() if not matched_symptoms.empty else 0

# Model Prediction function with severity score
def get_predicted_value_with_severity(patient_symptoms):
    # Convert symptoms to model input vector
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    # Predict disease
    predicted_disease = diseases_list.get(svc.predict([input_vector])[0], "Unknown Disease")

    # Calculate severity score
    total_severity = calculate_severity(patient_symptoms)

    return predicted_disease, total_severity


# creating routes========================================


# Define a route for the home page
# Define route for the home page
@app.route("/")
def home():
    return render_template("home.html")  # This renders home.html
@app.route('/login')
def login():
    return render_template('login.html')
# Define route for the index page
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/game")
def game():
    return render_template("game.html")



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()

        if not symptoms or symptoms.lower() == "symptoms":
            return render_template('index.html', message="Please enter valid symptoms.")

        # Process symptoms: Replace spaces with underscores and sanitize input
        user_symptoms = [s.strip("[]' ").replace(" ", "_") for s in symptoms.split(',')]
        user_symptoms = [symptom for symptom in user_symptoms if symptom in symptoms_dict]

        if not user_symptoms:
            return render_template('index.html', message="Invalid symptoms entered. Please provide valid symptoms.")

        # Get predictions
        predicted_disease, severity_score = get_predicted_value_with_severity(user_symptoms)

        # Determine severity category
        if severity_score <= 7:
            severity_category, recommendation = "Mild", "Self-care and monitoring."
        elif 8 <= severity_score <= 15:
            severity_category, recommendation = "Moderate", "Consult a doctor if symptoms persist."
        else:
            severity_category, recommendation = "Severe", "Immediate medical consultation recommended."

        # Get additional details
        dis_des, precautions, medications, rec_diet, workout, severity, gemini_details = helper(predicted_disease, severity_score)

        return render_template(
            'index.html',
            predicted_disease=predicted_disease,
            severity_score=severity,
            severity_category=severity_category,
            recommendation=recommendation,
            dis_des=dis_des,
            my_precautions = precautions[0].tolist() if precautions else [],
            medications=medications,
            my_diet=rec_diet,
            workout=workout,
            gemini_details=gemini_details  # Display additional insights
        )

    return render_template('index.html')


@app.route('/ask_gemini', methods=['POST'])
def ask_gemini_route():
    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"error": "Empty question"}), 400

    ai_answer = ask_gemini(question)  # Function to get response from Gemini
    return jsonify({"answer": ai_answer})

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.health_data
symptoms_collection = db.symptoms

# Risk Prediction Logic
def predict_risk(symptoms_list):
    risk_mapping = {
        "fever": "Viral Infection",
        "cough": "Respiratory Issue",
        "chicken pox": "Contagious Disease",
        "shortness of breath": "Lung Problem",
        "fatigue": "Possible Chronic Condition",
        "flu": "Seasonal Illness",
        "chikungunya": "Viral Infection",
        "cold": "Seasonal Illness",
        "headache": "Migraine/Stress",
        "nausea": "Gastrointestinal Issue",
        "vomiting": "Gastrointestinal Issue",
        "dizziness": "Neurological Issue",
        "diarrhea": "Gastrointestinal Issue",
        "rash": "Skin Infection",
        "sore throat": "Respiratory Issue",
        "runny nose": "Respiratory Issue",
        "stomach pain": "Gastrointestinal Issue",
        "joint pain": "Arthritis or Musculoskeletal Issue",
        "chills": "Viral Infection",
        "sweating": "Infection or Hormonal Imbalance",
        "weight loss": "Metabolic Disorder",
        "swelling": "Inflammatory Response",
        "insomnia": "Sleep Disorder",
        "mood swings": "Mental Health Issue",
        "loss of appetite": "Possible Chronic Condition",
        "memory loss": "Cognitive Disorder",
        "high blood pressure": "Cardiovascular Disease",
        "chest pain": "Cardiovascular Disease",
        "back pain": "Musculoskeletal Issue",
        "skin discoloration": "Dermatological Condition",
        "numbness": "Neurological Issue",
        "vision problems": "Ophthalmological Issue",
        "hearing loss": "Otorhinolaryngological Issue",
        "fainting": "Cardiovascular Issue",
        "abdominal bloating": "Gastrointestinal Issue",
        "constipation": "Gastrointestinal Issue",
        "painful urination": "Urinary Tract Infection",
        "frequent urination": "Urinary Tract Infection",
        "blood in stool": "Gastrointestinal Issue",
        "anxiety": "Mental Health Issue",
        "depression": "Mental Health Issue",
        "palpitations": "Cardiac Issue",
        "coughing up blood": "Respiratory or Cardiovascular Issue",
        "swollen glands": "Infection or Immune Response",
        "frequent headaches": "Neurological Disorder",
        "sore muscles": "Musculoskeletal Strain",
        "tingling sensation": "Neurological Disorder"
    }

    risks = set()
    for symptom in symptoms_list:
        if symptom.lower() in risk_mapping:
            risks.add(risk_mapping[symptom.lower()])
    return list(risks) if risks else ["No Major Risks Detected"]

@app.route('/report')
def report():
    return render_template("report.html")

@app.route('/submit_symptoms', methods=['POST'])
def submit_symptoms():
    data = request.json  
    symptoms_collection.insert_one({"symptoms_history": data})
    return jsonify({"message": "Symptoms saved successfully!"})

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    records = list(symptoms_collection.find({}, {"_id": 0, "symptoms_history": 1}))
    return jsonify(list(records))

@app.route('/generate_report', methods=['GET'])
def generate_report():
    records = list(symptoms_collection.find({}, {"_id": 0, "symptoms_history": 1}))
    
    symptoms_count = {}
    all_symptoms = []

    for record in records:
        symptoms_history = record.get("symptoms_history", [])
        if isinstance(symptoms_history, list):
            for entry in symptoms_history:
                if isinstance(entry, dict) and "date" in entry and "symptoms" in entry:
                    symptom = entry["symptoms"]
                    all_symptoms.append(symptom)
                    symptoms_count[symptom] = symptoms_count.get(symptom, 0) + 1

    # Predict Risks
    predicted_risks = predict_risk(all_symptoms)

    # Generate a Graph and Save as File
    chart_path = "symptom_chart.png"
    plt.figure(figsize=(6, 4))
    plt.bar(symptoms_count.keys(), symptoms_count.values(), color='green')
    plt.xlabel("Symptoms")
    plt.ylabel("Occurrences")
    plt.title("Symptom Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_path)  # Save to file
    plt.close()  # Close to free memory

    # Create PDF
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 750, "Health Report")
    
    y_position = 730
    pdf.setFont("Helvetica", 12)
    
    # Symptom History Section
    pdf.drawString(100, y_position, "Symptom History:")
    y_position -= 20
    for record in records:
        symptoms_history = record.get("symptoms_history", [])
        if isinstance(symptoms_history, list):
            for entry in symptoms_history:
                if isinstance(entry, dict) and "date" in entry and "symptoms" in entry:
                    pdf.drawString(120, y_position, f"• {entry['date']}: {entry['symptoms']}")
                    y_position -= 20
                    if y_position < 100:
                        pdf.showPage()
                        y_position = 750

    # Predicted Risks Section
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, y_position, "Predicted Risks:")
    y_position -= 20
    pdf.setFont("Helvetica", 12)
    for risk in predicted_risks:
        pdf.drawString(120, y_position, f"• {risk}")
        y_position -= 20

    # Add New Page & Embed the Graph
    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, 750, "Symptom Frequency Chart")
    
    if os.path.exists(chart_path):  # Check if image exists before adding
        pdf.drawImage(chart_path, 100, 300, width=400, height=250)
    
    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="health_report.pdf", mimetype="application/pdf")

@app.route('/risk')
def risk():
    return render_template("risk.html")

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)