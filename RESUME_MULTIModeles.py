import streamlit as st
import json
import os
from transformers import MBartForConditionalGeneration, MBartTokenizer, T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
import torch

# Vérifier si CUDA est disponible et définir le périphérique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des différents modèles et tokenizers
def load_model(model_name):
    if model_name == "mBART":
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
    elif model_name == "T5":
        model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
    elif model_name == "BART":
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    elif model_name == "Pegasus":
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    return model, tokenizer

# Définir le fichier pour enregistrer l'historique des chats
HISTORIQUE_FILE = "historique.json"

# Fonction pour charger l'historique depuis un fichier JSON
def charger_historique():
    if os.path.exists(HISTORIQUE_FILE):
        with open(HISTORIQUE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Fonction pour enregistrer l'historique dans un fichier JSON
def enregistrer_historique(historique):
    with open(HISTORIQUE_FILE, "w", encoding="utf-8") as file:
        json.dump(historique, file, ensure_ascii=False, indent=4)

# Initialiser l'historique depuis le fichier JSON
if "historique" not in st.session_state:
    st.session_state.historique = charger_historique()

if "texte" not in st.session_state:
    st.session_state.texte = ""

# Interface utilisateur
st.title("Résumé de Texte Multilingue avec Plusieurs Modèles")

# 📌 BARRE LATÉRALE
with st.sidebar:
    st.header("⚙️ Options")

    # Sélectionner le modèle
    model_choice = st.selectbox("Choisissez le modèle", ["mBART", "T5", "BART", "Pegasus"])

    # Charger le modèle et tokenizer sélectionnés
    model, tokenizer = load_model(model_choice)

    # Bouton pour démarrer une nouvelle conversation
    if st.button("🆕 Nouveau Chat"):
        st.session_state.texte = ""  # Réinitialiser seulement le texte de la zone de saisie
        st.rerun()

    # Bouton pour supprimer uniquement l'historique des conversations
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.historique = []
        enregistrer_historique([])  # Effacer l'historique du fichier JSON
        st.rerun()

    # Affichage de l'historique des conversations dans la barre latérale
    st.subheader("📜 Historique des Conversations")
    if st.session_state.historique:
        for index, conv in enumerate(st.session_state.historique):
            with st.expander(f"📌 Conversation {index + 1}", expanded=False):
                st.write("**📝 Texte Original :**")
                st.write(conv["original"])
                st.write("**📖 Résumé :**")
                st.write(conv["resume"])

                # Bouton pour supprimer une conversation spécifique
                if st.button(f"❌ Supprimer la conversation {index + 1}", key=f"del_{index}"):
                    st.session_state.historique.pop(index)
                    enregistrer_historique(st.session_state.historique)  # Mettre à jour le fichier
                    st.rerun()  # Rafraîchir l'interface
    else:
        st.info("Aucune conversation enregistrée.")

# Sélecteur de langue
langue = st.selectbox("Choisissez la langue du texte :", ["français", "anglais", "arabe"])

# Déterminer le code de langue pour mBART
langue_codes = {"français": "fr_XX", "anglais": "en_XX", "arabe": "ar_AR"}
tgt_lang = langue_codes[langue]

# Zone de texte pour l'entrée utilisateur (liée à session_state)
st.session_state.texte = st.text_area(f"Entrez le texte à résumer ({langue}) :", st.session_state.texte)

# Sélecteur de longueur de résumé
max_length = st.slider("Longueur maximale du résumé", min_value=50, max_value=200, value=130)
min_length = st.slider("Longueur minimale du résumé", min_value=20, max_value=100, value=50)

# Bouton pour générer le résumé
if st.button("📜 Générer le résumé"):
    if st.session_state.texte.strip():  # Vérifier si l'utilisateur a bien entré du texte
        # Prétraiter le texte
        inputs = tokenizer(st.session_state.texte.strip(), return_tensors="pt", max_length=1024, truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        # Générer le résumé
        if model_choice == "mBART":
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.0,
                num_beams=5,
                do_sample=False,
                temperature=0.7,
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]  # Définir la langue cible pour mBART
            )
        else:
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.0,
                num_beams=5,
                do_sample=False,
                temperature=0.7,
                early_stopping=True
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Sauvegarder la conversation dans l'historique
        st.session_state.historique.append({"original": st.session_state.texte, "resume": summary})
        enregistrer_historique(st.session_state.historique)  # Enregistrer l'historique dans le fichier

        # Affichage du résumé
        st.subheader("📖 Texte Résumé :")
        st.write(summary)
    else:
        st.warning("Veuillez entrer un texte à résumer.")