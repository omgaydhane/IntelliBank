from django.shortcuts import render, redirect
import csv
import os
import pandas as pd
import plotly.express as px 
from plotly.offline import plot
from pandasai import PandasAI
from api.models import User, Expert, Docs
from api.serializers import UserSerializer
from hashlib import sha256
from django.core.mail import send_mail
from django.core.mail import EmailMessage
from django.http import HttpResponse
from core.settings import EMAIL_HOST_USER
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from functools import lru_cache
import joblib
import pickle

adb_model_path = "models/text_classifier_with_threshold.pkl"
vectorizer_path = "models/tfidf_vectorizer_all.pkl"
adb_model = joblib.load(adb_model_path)
vectorizer = joblib.load(vectorizer_path)
model_path = "models/model.pkl"
clustervect = "models/tfidf_vectorizer_clustering.pkl"
kmean = "models/kmeans_clustering_model.pkl"
with open(clustervect, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open(kmean, 'rb') as file:
    kmeans_model = pickle.load(file)

def generate_label_from_keywords(keywords):
    return '_'.join(keywords)

def extract_keywords_and_generate_label(input_text, n_keywords=3):

    vectorized_input = tfidf_vectorizer.transform([input_text])
    predicted_cluster = kmeans_model.predict(vectorized_input)[0]
    similarity = cosine_similarity(vectorized_input, kmeans_model.cluster_centers_[predicted_cluster].reshape(1, -1))
    
    if similarity < 0.5:
        input_keywords = vectorized_input.toarray().flatten().argsort()[-n_keywords:][::-1]
        top_keywords = [tfidf_vectorizer.get_feature_names_out()[i] for i in input_keywords]
        label = "Open Intent: " + generate_label_from_keywords(top_keywords)
    else:
        centroid_keywords = kmeans_model.cluster_centers_[predicted_cluster].argsort()[-n_keywords:][::-1]
        top_keywords = [tfidf_vectorizer.get_feature_names_out()[i] for i in centroid_keywords]
        label = f"Cluster {predicted_cluster}: " + generate_label_from_keywords(top_keywords)

    return label

threshold = 0.2

def combined_pipeline(input_text):
    # First, use the simple classifier
    transformed_text = vectorizer.transform([input_text])
    probabilities = adb_model.predict_proba(transformed_text)
    max_probability = np.max(probabilities)
    predicted_class = adb_model.classes_[np.argmax(probabilities)]

    if max_probability >= threshold and predicted_class != "Open Intent":
        return f"Predicted category by ADB: {predicted_class}"
    else:
        print("Detected as Open Intent by ADB. Further analyzing with MTP-CLNN...")
        predict = extract_keywords_and_generate_label(input_text)
        return f"Detected as Open Intent by ADB. Further analyzing with MTP-CLNN. Intent: {predict}"
    
def predict(request):
    # Check if this is a POST request
    if request.method == 'POST':
        # Get user input from POST request
        user_input = request.POST.get('user_input')

        # Ensure user_input is not None or empty
        if user_input:
            # Make prediction (modify this according to your model's input format)
            prediction = combined_pipeline(user_input)

            # Render a template with the prediction or return it directly
            return render(request, 'result.html', {'prediction': prediction})
        else:
            # Redirect back to form or display an error message if input is invalid
            return redirect('predict')

    # For GET requests, just show the form
    return render(request, 'predict.html')

def index(request):
    if request.session.get('user_id'): return redirect(home)
    msg = {}
    msg["title"] = "Welcome"
    return render(request, 'index.html', msg)

def login(request):
    if request.session.get('user_id'):
        return redirect(home)
    msg = {}
    msg["title"] = "Login"
    msg["status"] = 1
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        hash = sha256(password.encode()).hexdigest()
        users = User.objects.filter(email=email)
        flg = 0
        for user in users:
            if user.password == hash:
                otp = str(random.randrange(1000,99999))
                otp_hash = str(sha256(otp.encode()).hexdigest())
                request.session['on_hold'] = user.id
                user.data = otp_hash
                user.save()
                flg = 1
                break
        if flg == 1:
            try:
                subject = "One-Time Password to Login to Your LegalEase Account"
                message = f"Hello {user.username} \n Your Verification OTP is : {otp}. \n Please use the OTP code to complete your login request.\n\n\n Best Regards,\n LegalEase"
                send_mail(subject, message, EMAIL_HOST_USER, [user.email], fail_silently=True)
                return redirect(email_verify)
            except:
                msg['status'] = 0
        else: msg['status'] = -1

    return render(request, 'login.html', msg)


def email_verify(request):
    if not request.session.get('on_hold'): return redirect(index)
    if request.session.get('user_id'): return redirect(home)
    user = User.objects.get(id=request.session['on_hold'])
    user_data = UserSerializer(user)
    msg = {}
    msg['username'] = user_data['username'].value
    msg['status'] = 1
    msg['title'] = 'Verify Email'
    if request.method == 'POST':
        otp = str(request.POST.get('otp'))
        hash = str(sha256(otp.encode()).hexdigest())
        if user_data['data'].value == hash:
            del request.session['on_hold']
            request.session['user_id'] = user_data['id'].value
            return redirect(index)
        msg['status'] = -1

    return render(request, 'emailverify.html', msg)
        

def register(request):
    if request.session.get('user_id'):
        return redirect(home)
    msg = {}
    msg["title"] = "Register"
    if request.method == 'POST':
        username, email, password = request.POST.get('username'), request.POST.get('email'), request.POST.get('password')
        hash = sha256(str(password).encode()).hexdigest()
        user = User(username=username, email=email, password=hash, data='')
        user.save()
        msg['status'] = 1
    return render(request, 'register.html', msg)


def logout(request):
    if request.session.get('user_id'):
        del request.session['user_id']
    return redirect(index)




@lru_cache()
def home(request):
    if not request.session.get('user_id'): return redirect(index)
    msg = {"title": "Dashboard", "description": "This is the landing Page"}
    number_of_docs = len(Docs.objects.all())
    msg["number_of_docs"] = number_of_docs
    msg["number_of_domains"] = 5
    msg["number_of_languages_supported"] = 25
   

   
    return render(request, 'home.html', msg)


