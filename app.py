import streamlit as st
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig

# 1. Configuration de l'interface
st.set_page_config(page_title="Project Sentinel - MedGemma 4B", layout="wide")
st.title("🏥 Project Sentinel : Assistant de Santé Mobile")
st.markdown("Numérisation locale et sécurisée via **MedGemma 1.5 4B**")

# 2. Chargement optimisé du modèle 4B
@st.cache_resource
def load_medgemma_4b():
    model_id = "google/medgemma-1.5-4b-it"
    
    # Configuration légère pour GPU domestique ou serveur cloud standard
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs={"quantization_config": bnb_config, "device_map": "auto"}
    )
    return pipe

# Gestion du chargement
try:
    with st.spinner("Initialisation de l'IA médicale (Version 4B)..."):
        vlm_pipe = load_medgemma_4b()
    st.sidebar.success("Modèle 4B opérationnel")
except Exception as e:
    st.sidebar.error("Note : Le modèle nécessite un GPU pour l'inférence réelle.")
    st.stop()

# 3. Interface utilisateur
uploaded_file = st.file_uploader("Prenez une photo du registre mensuel", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Document source", use_container_width=True)

    with col2:
        if st.button("Extraire les statistiques"):
            # Prompt spécifique pour le modèle 4B (direct et structuré)
            prompt = "Analyse ce registre médical. Extrais les colonnes Date, Diagnostic et Traitement sous forme de tableau JSON. Calcule le total par maladie."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            with st.spinner("Extraction en cours..."):
                outputs = vlm_pipe(text=messages, max_new_tokens=400)
                # Extraction du texte généré
                response = outputs[0]["generated_text"][-1]["content"]
                
                st.write("### Rapport Généré :")
                st.markdown(response)

st.divider()
st.info("💡 **Avantage 4B :** Ce modèle peut être déployé sur un ordinateur portable de milieu de gamme sans connexion internet.")