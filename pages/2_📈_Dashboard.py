import streamlit as st
import pickle
import altair as alt
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#===================================================================================================#
#                                   CONFIG INITIALE DE L'APP                                        #
#===================================================================================================#
# Configurer un titre, une icône et un layout pour l'application
st.set_page_config(page_title="P7 - POC", page_icon="📈", layout="centered", initial_sidebar_state="auto")

#===================================================================================================#

models_list = ['M1_not_augmented', 'M2_manually_augmented', 'M3_rand_augmented']

def load_model(model_name):
  models_directory = "./models/"
  model_path = f"{models_directory}/{model_name}.pkl"
  # Charger les données sauvegardées avec Pickle
  with open(model_path, 'rb') as f:
      model_data = pickle.load(f)
  return model_data

with st.sidebar:
    st.title('P7 - POC - RandAugment')
    selected_model = st.selectbox('Select model', models_list, index=0)

    photos_list = ['chihuahua', 'malinois', 'malamute']
    selected_photo = st.radio('Select photo', photos_list, index=0)

    if selected_photo is not None:
        st.image(f'./images/races/{selected_photo}.jpg', width=300)

with st.container():
    st.header('📈 Dashboard')
    # Recuperations des informations du modeles
    race_name = selected_photo.split('_')[0].lower()
    model_data = load_model(selected_model)
    model = model_data['model']
    model_history = model_data['model_history']
    model_predictions = model_data['predictions']
    model_confusion_matrix = model_data['confusion_matrix']
    model_class_name = model_data['class_name']

    # Recuperations des Metriques
    accuracy = round(model_history['accuracy'][-1], 4)
    loss = round(model_history['loss'][-1], 4)
    validation_accuracy = round(model_history['val_accuracy'][-1], 4)
    validation_loss = round(model_history['val_loss'][-1], 4)

    # Recuperations des Predictions
    top_prediction = model_predictions[race_name]['top_prediction']
    top_prediction_proba = model_predictions[race_name]['top_prediction_prob']
    # Créez un DataFrame pour les 3 prédictions
    top_3_predictions_df = pd.DataFrame(model_predictions[race_name]['all_predictions'])
    top_3_predictions = model_predictions[race_name]['all_predictions']
    other_predictions = '\n'.join([f"{i+1}) {pred['pred_class_name']} -> {pred['pred_class_name_prob']}" for i, pred in enumerate(top_3_predictions)])

    #===================================================================================================#
    #                                   AFFICHAGE DES RESULTATS                                         #
    #===================================================================================================#
    
    # AFFICHER LES PREDICTIONS
    st.header('Predictions')

    # Création de 5 colonnes
    col1, col2, col3, col4, col5 = st.columns([1, 0.5, 1.5, 0.5, 1])

    # Placer les images dans chaque colonne
    with col1:
        st.image(f"./images/races/{selected_photo}.jpg", caption=f"{selected_photo}", use_container_width=True)

    with col2:
        st.markdown('<span style="font-size:50px; color:green;">&#8594;</span>', unsafe_allow_html=True)

    with col3:
    # Utilisation de st.text avec du CSS pour centrer verticalement
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 100%; text-align: center; border: 2px solid grey; border-radius: 15px; padding: 10px;">
                <div style="text-align: center;">
                <h3>Modele : {selected_model}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown('<span style="font-size:50px; color:red;">&#8594;</span>', unsafe_allow_html=True)

    with col5:
        st.image(f'./images/races/{top_prediction}.jpg', caption=f"{top_prediction}", use_container_width=True)

    # AFFICHER LE TABLEAU DES PREDICTIONS
    
    st.text("Details des predictions :")
    prob1, prob2, prob3 = st.columns(3)
    prob1.metric(top_3_predictions[0]['pred_class_name'], f"{top_3_predictions[0]['pred_class_name_prob']:.6f}", border=True)
    prob2.metric(top_3_predictions[1]['pred_class_name'], f"{top_3_predictions[1]['pred_class_name_prob']:.6f}", border=True)
    prob3.metric(top_3_predictions[2]['pred_class_name'], f"{top_3_predictions[2]['pred_class_name_prob']:.6f}", border=True)
    

    # AFFICHER LES METRIQUES
    st.header(f'Metriques du modele : {selected_model}')
    a, b = st.columns(2)
    c, d = st.columns(2)
    
    a.metric("Accuracy", accuracy, f"{round(accuracy - validation_accuracy, 4)}", border=True)
    b.metric("Loss", loss, f"{round(loss - validation_loss, 4)}", border=True)
    c.metric("Validation Accuracy", validation_accuracy, f"{round(validation_accuracy - accuracy, 4)}", border=True)
    d.metric("Validation Loss", validation_loss, f"{round(validation_loss - loss, 4)}", border=True)

    # HISTORIQUE DES METRIQUES
    data = pd.DataFrame(model_history)
    st.header('Historique des metriques sur 10 epochs')
    st.line_chart(data, x_label="Epochs", y_label="Metrics", color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'], use_container_width=True)

    # CONFUSION MATRIX
    st.header('Matrice de confusion')
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model_class_name, yticklabels=model_class_name)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion')
    st.pyplot(plt)