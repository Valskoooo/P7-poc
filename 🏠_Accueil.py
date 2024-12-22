import streamlit as st

#===================================================================================================#
#                                   CONFIG INITIALE DE L'APP                                        #
#===================================================================================================#
# Configurer un titre, une icône et un layout pour l'application
st.set_page_config(page_title="P7 - POC", page_icon="🏠", layout="centered", initial_sidebar_state="auto")

#===================================================================================================#

with st.container():
    st.header('✍️ P7 - Preuve de concept - RandAugment')
    st.markdown('*Dumont Valentin - Copyright@2024*')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("./images/assets/poc.png", width=200)

    st.subheader('📚 Contexte du projet')
    st.text('''Ce projet intervient dans le contexte de la réalisation d'une preuve de concept pour le projet 7 du parcours Machine Learning Engineer d'OpenClassrooms. Il est la continuité du projet 6 intitulé "Classez des images à l'aide d'algorithmes de Deep Learning" et qui avait pour but la création d'un système de classification d'images de chien à partir d'un réseau de neurones convolutionnels.

Lors de la réalisation du projet 6, la phase de prétraitement des images incluant une data augmentation manuelle a été une étape cruciale pour améliorer les performances du modèle. Cependant, la data augmentation est une tâche fastidieuse et chronophage. Pour pallier à ce problème, nous avons décidé d'explorer une technique de data augmentation automatique appelée RandAugment.

RandAugment est une méthode d'augmentation de données pour l'entraînement des modèles d'apprentissage profond, principalement utilisée pour améliorer la généralisation des réseaux neuronaux, notamment dans les tâches de vision par ordinateur. Il a été introduit dans un article de recherche en 2019 et fait partie de la famille des techniques d'augmentation de données qui permettent de rendre les modèles plus robustes en générant artificiellement de nouvelles données à partir des données d'entraînement existantes.''')
    
    st.subheader('📈 Objectifs')
    st.text('''L'augmentation de données est une technique consistant à transformer les données d'entraînement pour créer de nouvelles variations de celles-ci. Cela permet de rendre les modèles plus robustes, surtout dans des scénarios où les données sont limitées. Avant l'introduction de RandAugment, de nombreuses approches d'augmentation étaient basées sur des stratégies plus complexes ou spécifiques aux types de transformations appliquées (comme dans AutoAugment, qui sélectionne les meilleures transformations à appliquer sur la base d'un processus de recherche).

RandAugment simplifie ce processus en ne nécessitant que deux hyperparamètres principaux et une sélection aléatoire parmi un ensemble d'opérations d'augmentation pré-définies. L'idée principale est de randomiser et de simplifier les transformations tout en obtenant une performance similaire à celle des méthodes plus complexes.

L'objectif de ce projet est de comparer les performances de trois modèles de réseaux de neurones convolutionnels entraînés sur des données augmentées de manière différente. Les trois modèles sont les suivants :
        - M1 : Modèle entraîné sur les données non augmentées
        - M2 : Modèle entraîné sur les données augmentées manuellement
        - M3 : Modèle entraîné sur les données augmentées avec RandAugment
''')
    
    st.subheader("🔧 Fonctionnement de RandAugment")
    st.text("""RandAugment se distingue par sa simplicité par rapport à d'autres méthodes d'augmentation de données comme AutoAugment, qui nécessitent une recherche coûteuse pour déterminer les meilleurs paramètres. Au lieu de cela, RandAugment utilise un nombre fixe d'opérations et choisit de manière aléatoire parmi celles-ci, en appliquant chaque transformation avec une magnitude spécifiée par un hyperparamètre.

Processus général de RandAugment :
Sélection d'opérations d'augmentation : Un ensemble d'opérations pré-définies (comme la rotation, le recadrage, l'inversion, la saturation, etc.) est utilisé.
Choix de l'opération : Une opération est choisie au hasard parmi cet ensemble.
Appliquer l'opération : Chaque transformation a un facteur de "magnitude" qui détermine l'intensité de la transformation appliquée.
Itérer : Ce processus est répété plusieurs fois pour chaque image dans le jeu de données d'entraînement.
            
Il existe principalement deux hyperparamètres clés qui contrôlent le fonctionnement de RandAugment :

N (le nombre d'opérations d'augmentation à appliquer): C'est le nombre total d'opérations qui seront appliquées à chaque image. Si N = 1, une seule transformation est appliquée à chaque image. Si N > 1, plusieurs transformations seront effectuées.
M (la magnitude des transformations): C'est le degré de transformation appliqué à l'image. Ce paramètre contrôle l'intensité de chaque opération. Par exemple, pour une opération de rotation, une faible magnitude pourrait signifier une rotation de quelques degrés, tandis qu'une grande magnitude pourrait entraîner une rotation plus importante.""")
    
    st.image("./images/assets/SCR-20241219-pfpz.jpeg", width=500, use_container_width=True)

    st.subheader("📝 Méthodologie")

    st.text("""Vous pouvez retrouver le plan prévisionnel ainsi que la note méthodologique du projet en cliquant sur les liens ci-dessous :""")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.text("📅 Plan prévisionnel")
        with open("./documents/Modèle_Plan_prévisionnel_RandAugment.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Télécharger le plan prévisionnel",
                            data=PDFbyte,
                            file_name="test.pdf",
                            mime='application/octet-stream')

    with col2:
        st.text("📄 Note méthodologique")
        with open("./documents/Modèle_Plan_prévisionnel_RandAugment.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Télécharger la note méthodologique",
                            data=PDFbyte,
                            file_name="test.pdf",
                            mime='application/octet-stream')