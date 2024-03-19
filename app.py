from flask import Flask, request, render_template
import pandas as pd
import openai
from openai import OpenAI

import configparser
import numpy as np

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('C:/Users/Boubker/Documents/IA Generative/Generative-AI-Module-Dauphine/config.ini') # Chemin vers votre fichier de configuration
OPENAI_KEY = config.get('OPENAI_API', 'OPENAI_KEY')
client = OpenAI(api_key=OPENAI_KEY)

file_path = "C:/Users/Boubker/Documents/IA Generative/Generative-AI-Module-Dauphine/data/twitter_light.csv"
data = pd.read_csv(file_path, sep=";",encoding='ISO-8859-1')


# Fonction pour obtenir l'embedding d'un texte
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Prétraitement pour calculer les embeddings pour chaque question dans le DataFrame data
def preprocess_data(data):
    # Appliquer get_embedding à chaque ligne du DataFrame
    data['customer_embed'] = data['customer_tweet'].apply(lambda x: get_embedding(x))
    return data

# Fonction pour calculer la similarité cosinus entre deux vecteurs
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    return dot_product / (magnitude_A * magnitude_B)

# Prétraitement des données et calcul des embeddings
#data= data[data['company'].str.contains('AmazonHelp', na=False)]
data = preprocess_data(data)

# Fonction pour trouver la question la plus similaire et renvoyer la réponse associée
def find_similar_question(embedding):
    similarities = []
    for index, row in data.iterrows():
        similarity = cosine_similarity(embedding, row['customer_embed'])
        similarities.append(similarity)
    max_similarity_index = similarities.index(max(similarities))
    return data.iloc[max_similarity_index]['company_tweet']



@app.route('/')
def index():
    return render_template('montemplate.html')

@app.route('/poser_question', methods=['POST'])
def poser_question():
    if request.method == 'POST':
        # Récupérer la question soumise par le formulaire
        question = request.form['question']
        embedding = get_embedding(question)
        print(find_similar_question(embedding))
        # Répondre à la question
        reponse=find_similar_question(embedding)
        # Retourner la réponse
        return reponse

if __name__ == '__main__':
    app.run(debug=True)





   