import streamlit as st
import pickle
import altair as alt
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from torchvision.transforms import RandAugment

#===================================================================================================#
#                                   CONFIG INITIALE DE L'APP                                        #
#===================================================================================================#
# Configurer un titre, une icône et un layout pour l'application
st.set_page_config(page_title="P7 - POC", page_icon="📚", layout="centered", initial_sidebar_state="auto")

#===================================================================================================#

with st.container():
    st.header('📚 Informations sur le jeu de données')
    st.subheader('📌 Description')
    st.text("""
    Le Stanford Dog Dataset est un ensemble de données contenant des images de chiens de 120 races différentes. Il a été créé pour des tâches de classification d'images et contient plus de 20 580 images au total. Chaque image est étiquetée avec la race du chien, ce qui permet de l'utiliser pour entraîner des modèles de reconnaissance d'images.

    Créateur: Stanford University
    Date de création: 2011
    Source: http://vision.stanford.edu/aditya86/ImageNetDogs/
    """)

    st.text("""Le dataset contient 20 580 images réparties en 120 races, soit en moyenne 171.5 images par race. Les images sont de tailles variables et en couleurs.""")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric(label="Nombre de races", value=120)

    with col2:
        st.metric(label="Nombre total d'images", value=20580)

    with col3:
        st.metric(label="Nombre moyen d'images par race", value=20580/120)

    st.image("./images/assets/distribution-empirique.png", width=500, use_container_width=True)

    st.subheader('📌 Sélection des races pour la modélisation')
    st.text("""
    Pour faciliter la modélisation des différents modèles utilisés, seules 3 races ont été sélectionnées : Chihuahua, Malinois et Malamute. Le dataset contient 152 images de Chihuahua, 150 images de Malinois et 178 images de Malamute. Cela simplifie le processus de modélisation et permet de se concentrer sur des résultats plus précis et spécifiques.

    La data augmentation est donc importante pour garantir des prédictions correctes et améliorer la robustesse du modèle. En augmentant artificiellement la taille du dataset avec des transformations telles que la rotation, le recadrage et le changement de luminosité, le modèle peut mieux généraliser et être plus performant sur des images inédites.
    """)

    # Récupérer les chemins des images dans les sous-dossiers
    image_folder = "./images/dataset/original/"
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # Limiter à 9 images pour l'affichage
    image_paths = image_paths[:9]

    # Créer un subplot pour afficher les images
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for img_path, ax in zip(image_paths, axes):
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    # Ajouter un titre au plot
    fig.suptitle('Exemples d\'images du dataset', fontsize=16)
    st.pyplot(fig)

    st.subheader("📸 Exemple d'images du dataset avec RandAugment")
    st.text("Pour améliorer la diversité des données d'entraînement, nous appliquons RandAugment, une technique de data augmentation qui applique des transformations aléatoires aux images.")

    import torchvision.transforms as transforms

    # Fonction pour appliquer RandAugment
    def apply_randaugment(image, num_ops=5, magnitude=10):
        transform = transforms.Compose([
            RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])
        return transform(image)

    # Récupérer les chemins des images dans les sous-dossiers
    image_folder = "./images/dataset/original/"
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # Limiter à 9 images pour l'affichage
    image_paths = image_paths[:9]

    # Créer un subplot pour afficher les images
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for img_path, ax in zip(image_paths, axes):
        img = Image.open(img_path)
        img_aug = apply_randaugment(img, num_ops=5, magnitude=7)
        ax.imshow(img_aug, cmap='gray')
        ax.axis('off')

    # Ajouter un titre au plot
    fig.suptitle('Exemples d\'images du dataset avec RandAugment', fontsize=16)
    st.pyplot(fig)
