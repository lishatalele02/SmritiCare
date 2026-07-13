from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')

# In-memory user storage (replace with a database in production)
users = {}

app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.tflite")

# Lazy-load the TFLite model to reduce startup time on Render
interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter, input_details, output_details
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

# Class labels for predictions
class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Recommendations for each stage
recommendations = {
    "MildDemented": {
        "doctor": "Dr. Smith, Neurologist",
        "medicines": "Donepezil, Memantine",
        "precautions": "Regular exercise, healthy diet, brain exercises",
        "description": "Mild Dementia means there are some memory problems, but the person can still do daily activities with little help."
    },
    "ModerateDemented": {
        "doctor": "Dr. Johnson, Dementia Specialist",
        "medicines": "Rivastigmine, Galantamine",
        "precautions": "Needs support for daily tasks, follow medication, caregiver help",
        "description": "Moderate Dementia means memory loss is worse, and the person needs help with daily activities like cooking or remembering things."
    },
    "NonDemented": {
        "doctor": "No specialist needed",
        "medicines": "No medication required",
        "precautions": "Eat healthy, exercise, and keep the brain active",
        "description": "Non-Demented means the brain is healthy, and there are no signs of memory loss or dementia."
    },
    "VeryMildDemented": {
        "doctor": "Dr. Brown, Cognitive Therapist",
        "medicines": "Vitamin B12, Omega-3",
        "precautions": "Do memory exercises, visit a doctor for checkups",
        "description": "Very Mild Dementia means there are small memory issues, but they do not affect daily life much."
    }
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def process_image(image_path):

    load_model()

    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img, dtype=np.float32)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_labels[np.argmax(prediction)]

    return {
        "stage": predicted_class,
        "doctor": recommendations[predicted_class]["doctor"],
        "medicines": recommendations[predicted_class]["medicines"],
        "precautions": recommendations[predicted_class]["precautions"],
        "description": recommendations[predicted_class]["description"]
    }
# Routes
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form.get('name', 'Unknown')
        contact = request.form.get('contact', 'Unknown')

        if email in users:
            return "User already exists! Try logging in.", 400
        elif password != confirm_password:
            return "Passwords do not match! Try again.", 400
        else:
            # Save user data
            users[email] = {
                'password': password,
                'name': name,
                'contact': contact,
                'report': 'No report available',
                'appointments': [],
                'reports': []
            }
            flash('You have to login', 'info')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')

    if email in users and users[email]['password'] == password:
        session['email'] = email
        flash('Login successful', 'success')
        return redirect(url_for('index'))
    else:
        flash('Invalid credentials! Try again.', 'error')
        return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Diagnosis')
@login_required  # Use the login_required decorator

def index():
    return render_template('index.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']
    if email in users:
        # Fetch user data from the users dictionary
        user_data = users[email]
        patient_data = {
            'patient_name': user_data.get('name', 'Unknown'),
            'patient_email': email,
            'patient_contact': user_data.get('contact', 'Unknown'),
            'patient_report': user_data.get('report', 'No report available'),
            'appointments': user_data.get('appointments', []),  # Pass appointments
            'reports': user_data.get('reports', [])  # Pass reports
        }
    else:
        # Default data if user is not found
        patient_data = {
            'patient_name': 'Unknown',
            'patient_email': email,
            'patient_contact': 'Unknown',
            'patient_report': 'No report available',
            'appointments': [],  # Default empty list for appointments
            'reports': []  # Default empty list for reports
        }

    return render_template('dashboard.html', **patient_data)

from flask import session, flash, redirect, url_for

@app.route('/logout')
def logout():
    
    session.pop('email', None)
    session.pop('_flashes', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        name = request.form.get('name', 'Unknown')
        age = request.form.get('age', 'Unknown')
        gender = request.form.get('gender', 'Unknown')

        results = process_image(filepath)

        # Keep uploaded image until the report page has been served.
        # You can remove it later with a cleanup job if desired.

        if 'email' in session:
            email = session['email']
            if email in users:
                users[email]['report'] = f"Alzheimer's Stage: {results['stage']}"

        return redirect(url_for(
            'report',
            name=name,
            age=age,
            gender=gender,
            mri_file=filename,
            stage=results['stage'],
            doctor=results['doctor'],
            medicines=results['medicines'],
            precautions=results['precautions'],
            description=results['description']
        ))

    except Exception as e:
        print("Prediction Error:", e)
        return f"Prediction Error: {e}", 500

@app.route('/report')
def report():
    name = request.args.get('name', 'Unknown')
    age = request.args.get('age', 'Unknown')
    gender = request.args.get('gender', 'Unknown')
    mri_file = request.args.get('mri_file', 'Unknown')
    stage = request.args.get('stage', 'Unknown')
    doctor = request.args.get('doctor', 'Unknown')
    medicines = request.args.get('medicines', 'Unknown')
    precautions = request.args.get('precautions', 'Unknown')
    description = request.args.get('description', 'No description available')  

    return render_template('report.html', 
                           name=name, 
                           age=age, 
                           gender=gender, 
                           mri_file=mri_file, 
                           stage=stage, 
                           doctor=doctor, 
                           medicines=medicines, 
                           precautions=precautions, 
                           description=description)  # Pass to template

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    email = session['email']
    data = request.get_json()
    name = data.get('name')
    contact = data.get('contact')

    if email in users:
        users[email]['name'] = name
        users[email]['contact'] = contact
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    else:
        return jsonify({'success': False, 'message': 'User not found'})

@app.route('/schedule_appointment', methods=['POST'])
def schedule_appointment():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    email = session['email']
    data = request.get_json()
    date = data.get('date')
    time = data.get('time')
    doctor = data.get('doctor')

    if email in users:
        if 'appointments' not in users[email]:
            users[email]['appointments'] = []
        users[email]['appointments'].append({'date': date, 'time': time, 'doctor': doctor})
        print(f"Appointments for {email}: {users[email]['appointments']}")  # Debug statement
        return jsonify({'success': True, 'message': 'Appointment scheduled successfully'})
    else:
        return jsonify({'success': False, 'message': 'User not found'})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'})

    email = session['email']
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if email in users:
        if 'reports' not in users[email]:
            users[email]['reports'] = []
        users[email]['reports'].append({'name': filename, 'date': '2023-10-10', 'link': f'/static/uploads/{filename}'})
        print(f"Reports for {email}: {users[email]['reports']}")  # Debug statement
        return jsonify({'success': True, 'message': 'Document uploaded successfully'})
    else:
        return jsonify({'success': False, 'message': 'User not found'})

if __name__ == '__main__':
    app.run(debug=True)