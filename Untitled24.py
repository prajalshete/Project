#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the dataset (example)
data = pd.read_csv('C:\\Users\\Dell\\Downloads\\medical_history1.csv')
data.head(20)


# In[3]:


data.isnull().sum()


# In[4]:


data.fillna(0)


# In[5]:


data.isnull().sum()


# In[6]:


# Fill NaN values in 'symptom' column with an empty string
data['symptom'].fillna('', inplace=True)


# In[7]:


data.dtypes


# In[8]:


# Data preprocessing (cleaning and feature extraction)
# You may need to customize this based on your dataset
#data['input'] = data['symptom'] + ' ' + data['Lifestyle']+data['Medical_History']


# In[9]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Tokenization and removing stopwords
stop_words = set(stopwords.words('english'))

#def text_preprocess(text):
  #  words = word_tokenize(text.lower())
   # words = [word for word in words if word.isalnum() and word not in stop_words]
    #return ' '.join(words)

def text_preprocess(text):
    if pd.notna(text):  # Check if the value is not NaN
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        return ' '.join(words)
    else:
        return ''  # Return an empty string for NaN values

data['symptom'] = data['symptom'].apply(text_preprocess)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['symptom'])
# Combine the text columns into a single column
#data['combined_text'] = data['symptom'] + ' ' + data['Lifestyle'] + ' ' + data['Medical_History']

# Separate features (X) and target variable (y)
#X = data['combined_text']
#y = data['Disease']


# In[11]:


#user_input = input("Enter your symptoms: ")
#user_input = input("Enter your lifestyle: ")
#user_input = input("Enter your Medica_History: ")
#user_input = text_preprocess(user_input)
#user_input = tfidf_vectorizer.transform([user_input])
#predicted_disease = classifier.predict(user_input)
#print("Predicted Disease:", predicted_disease[0])


# In[12]:


# Function to predict disease based on symptoms
def get_disease(user_input, data):
    # Preprocess user input
    user_input =text_preprocess(user_input)

    # Filter the DataFrame to match the user's input symptom
    filtered_data = data[data['symptom'] == user_input]

    # predict disease for the input symptom
    disease = filtered_data['Disease']

    return disease

# Example usage for symptoms
user_input = input("Enter symptoms\n")
predictions = get_disease(user_input, data)

print("predict disease for", user_input, ":")
print(predictions)


# In[13]:


# Function to get medicine recommendations based on multiple symptoms
"""def get_medicine_recommendations(user_inputs, data):
    # Preprocess user inputs
    user_inputs = [text_preprocess(symptom) for symptom in user_inputs]

    # Filter the DataFrame to match the user's input symptoms
    filtered_data = data[data['symptom'].isin(user_inputs)]

    # Get recommended medications for the input symptoms
    recommended_medications = filtered_data['Disease'].unique()

    return recommended_medications

# Example usage for multiple symptoms
num_symptoms = int(input("How many symptoms do you want to enter?\n"))
user_inputs = [input(f"Enter symptom #{i+1}\n") for i in range(num_symptoms)]

recommendations = get_medicine_recommendations(user_inputs, data)

print("Predicted diseases for the given symptoms:")
print(recommendations)"""


# In[18]:


# Function to get medicine recommendations based on multiple symptoms, lifestyle, and medical history
"""def get_medicine_recommendations(user_inputs, Lifestyle, Medical_History, data):
    # Preprocess user inputs
    user_inputs = [text_preprocess(symptom) for symptom in user_inputs]
    lifestyle = text_preprocess(Lifestyle)
    medical_history = text_preprocess(Medical_History)

    # Filter the DataFrame to match the user's input symptoms, lifestyle, and medical history
    filtered_data = data[data['symptom'].isin(user_inputs) & (data['Lifestyle'] == Lifestyle) & (data['Medical_History'] == Medical_History)]
# Debugging: Print the filtered data
    #print("Filtered Data:")
    #print(filtered_data)
    # Get recommended medications for the input symptoms, lifestyle, and medical history
    recommended_medications = filtered_data['Disease'].unique()

    return recommended_medications

# Example usage for multiple symptoms, lifestyle, and medical history
#num_symptoms = int(input("How many symptoms do you want to enter?\n"))
#user_inputs = [input(f"Enter symptom #{i+1}\n") for i in range(num_symptoms)]
user_inputs = input("Enter symptoms\n")

lifestyle = input("Enter lifestyle information\n")
medical_history = input("Enter medical history information\n")

recommendations = get_medicine_recommendations(user_inputs, lifestyle, medical_history, data)

print("Predicted diseases for the given symptoms, lifestyle, and medical history:")
print(recommendations)"""


# In[ ]:


"""import streamlit as st
# Train the model
# Define and initialize the classifier
#classifier = RandomForestClassifier()

# Use the classifier in your code
#classifier.fit(X_train, y_train)

# Streamlit App
st.title("Disease Prediction App")

# User input
user_input = st.text_area("Enter your symptoms and lifestyle:")

if st.button("Predict"):
    # Preprocess user input
    user_input = text_preprocess(user_input)
    user_input = tfidf_vectorizer.transform([user_input])

    # Make prediction
    predicted_disease = classifier.predict(user_input)

    # Display result
    st.success(f"Predicted Disease: {predicted_disease[0]}")"""


# In[ ]:


import streamlit as st

# Streamlit web app title and description
st.title("Disease Prediction App")
#st.markdown("Enter your symptoms, and we'll recommend medications.")

# Input for user's symptoms
user_input = st.text_area("Enter your symptoms:")

# Button to generate recommendations
if st.button("Predict"):
    # Split user input into a list of symptoms
    user_symptoms = [s.strip() for s in user_input.split()]

    # Check if the input is not empty
    if user_symptoms:
        # Get recommendations (replace with your recommendation logic)
        disease = get_disease(" ".join(user_symptoms), data)

        # Display recommendations
        st.subheader("Predict Disease:")
        # Display the list of recommended medications
        for prediction in disease:
            st.write(prediction)


# In[ ]:




