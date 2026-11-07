"""SupportBot : l'application de démonstration du chatbot (streamlit run app/app.py)."""

import sys
from pathlib import Path

from urllib.parse import urlparse

import streamlit as st
import torch

torch.classes.__path__ = []  # évite l'avertissement de Streamlit sur torch.classes

RACINE = Path(__file__).resolve().parent.parent
sys.path.append(str(RACINE / "src"))
from supportbot import SupportBot  # noqa: E402

# ---------------------------------------------------------------------------
# Marques proposées : nom affiché, et page d'aide qui remplace <url>
# ---------------------------------------------------------------------------
MARQUES = {
    "AppleSupport": ("Apple", "https://support.apple.com"),
    "AmazonHelp": ("Amazon", "https://www.amazon.com/gp/help/customer/display.html"),
    "SpotifyCares": ("Spotify", "https://support.spotify.com"),
    "Uber_Support": ("Uber", "https://help.uber.com"),
    "AskPayPal": ("PayPal", "https://www.paypal.com/smarthelp/home"),
    "Delta": ("Delta Air Lines", "https://www.delta.com"),
    "AmericanAir": ("American Airlines", "https://www.aa.com"),
    "SouthwestAir": ("Southwest Airlines", "https://www.southwest.com"),
    "British_Airways": ("British Airways", "https://www.britishairways.com"),
    "VirginAtlantic": ("Virgin Atlantic", "https://www.virginatlantic.com"),
    "XboxSupport": ("Xbox", "https://support.xbox.com"),
    "hulu_support": ("Hulu", "https://help.hulu.com"),
    "comcastcares": ("Comcast Xfinity", "https://www.xfinity.com/support"),
    "TMobileHelp": ("T-Mobile", "https://www.t-mobile.com/support"),
    "UPSHelp": ("UPS", "https://www.ups.com"),
    "sainsburys": ("Sainsbury's", "https://help.sainsburys.co.uk"),
    "Tesco": ("Tesco", "https://www.tesco.com/help"),
    "DunkinDonuts": ("Dunkin'", "https://www.dunkindonuts.com"),
    "ChipotleTweets": ("Chipotle", "https://www.chipotle.com"),
}

# Exemples de messages typiques, adaptés à la marque choisie (pour la démonstration)
EXEMPLES = {
    "AppleSupport": ["my iphone battery drains really fast since the ios update",
                     "my airpods won't connect to my macbook anymore"],
    "AmazonHelp": ["my package says delivered but i never received it",
                   "i was charged twice for the same order"],
    "SpotifyCares": ["spotify keeps crashing on my android phone",
                     "i was charged twice for premium this month"],
    "Uber_Support": ["my driver took a longer route and i was overcharged",
                     "i left my phone in an uber last night"],
    "AskPayPal": ["my paypal account has been limited and i don't know why",
                  "i sent money to the wrong email address"],
    "Delta": ["my flight was delayed and i missed my connection",
              "my bag didn't arrive at my destination"],
    "AmericanAir": ["my flight was cancelled and nobody at the airport can help me",
                    "can i change my seat after booking ?"],
    "XboxSupport": ["i can't sign in to xbox live on my console",
                    "my controller keeps disconnecting"],
    "comcastcares": ["my internet has been down since this morning",
                     "why is my bill higher this month ?"],
}
EXEMPLES_GENERAUX = ["my order hasn't arrived yet and it's been two weeks!",
                     "i was charged twice this month",
                     "i can't log in to my account",
                     "your app keeps crashing since the last update"]

st.set_page_config(page_title="SupportBot", page_icon="💬", layout="centered")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 3.2rem; max-width: 820px; }

/* Bulles de la conversation */
[data-testid="stChatMessage"] { border-radius: 18px; padding: 0.9rem 1rem; margin-bottom: 0.7rem; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: rgba(29, 155, 240, 0.14); border: 1px solid rgba(29, 155, 240, 0.45);
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
    background: #192734; border: 1px solid #38444D; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}

/* Statut « en ligne » et conseils dans la barre latérale */
.statut {
    display: flex; align-items: center; gap: 0.5rem; padding: 0.55rem 0.8rem; margin-bottom: 1rem;
    border-radius: 999px; background: rgba(0, 186, 124, 0.10); border: 1px solid rgba(0, 186, 124, 0.45);
    color: #00BA7C; font-size: 0.85rem; font-weight: 600;
}
.point {
    width: 9px; height: 9px; border-radius: 50%; background: #00BA7C; flex-shrink: 0;
    animation: pulsation 1.6s ease-in-out infinite;
}
@keyframes pulsation { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.conseils div {
    background: #15202B; border: 1px solid #38444D; border-radius: 12px;
    padding: 0.55rem 0.75rem; margin-bottom: 0.45rem; font-size: 0.86rem; color: #F7F9F9;
}

/* Zone « Cette réponse vous a-t-elle aidé ? » */
.avis { color: #8B98A5; font-size: 0.85rem; margin-top: 0.6rem; margin-bottom: 0.1rem;
        padding-top: 0.5rem; border-top: 1px solid #38444D; }

/* Titre et sous-titre */
.titre, .titre p {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0; line-height: 1.2;
    background: linear-gradient(90deg, #1D9BF0 0%, #8ECDF8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sous-titre { color: #8B98A5; font-size: 1.02rem; margin-top: 0.2rem; margin-bottom: 1.4rem; }

/* Badges (marque, modèle, méthode) */
.badge {
    display: inline-block; padding: 0.2rem 0.75rem; margin: 0 0.35rem 0.35rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(29, 155, 240, 0.12); color: #1D9BF0; border: 1px solid rgba(29, 155, 240, 0.45);
}
.badge .logo {
    width: 16px; height: 16px; vertical-align: -3px; margin-right: 6px;
    border-radius: 4px; background: #ffffff; padding: 1px;
}

/* Cas similaire retrouvé */
.cas {
    border-left: 3px solid #00BA7C; background: rgba(0, 186, 124, 0.08); border-radius: 8px;
    padding: 0.7rem 0.9rem; font-size: 0.9rem; color: #F7F9F9; margin-top: 0.3rem;
}
.cas b { color: #00BA7C; }

/* Zone vide avant le premier message */
.vide {
    text-align: center; color: #8B98A5; padding: 2.2rem 1rem 1rem 1rem;
    border: 1px dashed #38444D; border-radius: 18px; margin-bottom: 1rem;
}
.vide .icone { font-size: 2.4rem; }

/* Boutons arrondis comme sur Twitter */
.stButton > button {
    border-radius: 999px; border: 1px solid #38444D; background: transparent; color: #F7F9F9;
}
.stButton > button:hover { border-color: #1D9BF0; color: #1D9BF0; background: rgba(29, 155, 240, 0.08); }

/* Barre latérale et zone de saisie */
[data-testid="stSidebar"] { background: #192734; border-right: 1px solid #38444D; }
[data-testid="stChatInput"] textarea { border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement des modèles…")
def charger_bot():
    return SupportBot(RACINE / "app" / "ressources")


bot = charger_bot()
marques_disponibles = [m for m in MARQUES if m in bot.numeros_marques]

# ---------------------------------------------------------------------------
# Barre latérale : réglages
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Réglages")
    marque = st.selectbox("Marque contactée", marques_disponibles,
                          format_func=lambda m: MARQUES[m][0])
    nom_modele = st.radio(
        "Modèle", ["Modèle 3", "Modèle 2", "Modèle 1"],
        captions=["Attention + cas similaire (RAG)", "LSTM avec attention", "LSTM (référence)"])
    NOMS_METHODES = {"beam": "Beam search (5)", "greedy": "Greedy", "creatif": "Créatif (échantillonnage)"}
    methode = st.radio("Génération", list(NOMS_METHODES), format_func=NOMS_METHODES.get)
    temperature = 0.8
    if methode == "creatif":
        temperature = st.slider("Température (variété des réponses)", 0.3, 1.5, 0.8, 0.1,
                                help="Plus elle est haute, plus les réponses sont variées, "
                                     "mais moins elles sont sûres.")
    st.divider()
    st.markdown('<div class="statut"><span class="point"></span>'
                'Assistant en ligne · réponse en quelques secondes</div>', unsafe_allow_html=True)
    st.markdown("### 💡 Conseils")
    st.markdown("""
<div class="conseils">
<div>✍️ Écrivez votre message en anglais, en une ou deux phrases.</div>
<div>📦 Précisez le produit ou le service concerné.</div>
<div>🔗 Suivez le lien « page d'aide » pour aller plus loin.</div>
<div>🔒 Ne partagez jamais votre mot de passe ni vos coordonnées bancaires.</div>
</div>
""", unsafe_allow_html=True)
    st.caption("Encodeur–décodeur LSTM construit et entraîné from scratch (PyTorch), "
               "sur 150 000 conversations de service client sur Twitter.")
    st.caption("Projet universitaire de démonstration, non affilié aux entreprises citées.")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.historique = []
        st.rerun()

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown('<div class="titre">SupportBot</div>', unsafe_allow_html=True)
st.markdown('<p class="sous-titre">Un chatbot de service client construit et entraîné from scratch : '
            'encodeur–décodeur LSTM, attention et recherche de cas similaires.</p>',
            unsafe_allow_html=True)
domaine = urlparse(MARQUES[marque][1]).netloc
logo = f"https://www.google.com/s2/favicons?domain={domaine}&sz=64"
st.markdown(f'<span class="badge"><img src="{logo}" class="logo">{MARQUES[marque][0]}</span>'
            f'<span class="badge">🧠 {nom_modele}</span>'
            f'<span class="badge">🔎 {NOMS_METHODES[methode]}</span>',
            unsafe_allow_html=True)


def mettre_en_forme(reponse, marque):
    """Majuscules en début de phrase, ponctuation recollée et <url> remplacé par un vrai lien."""
    texte = reponse.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
    phrases = [p.strip().capitalize() for p in texte.split(". ")]
    texte = ". ".join(phrases).replace(" i ", " I ").replace(" i'", " I'")
    lien = MARQUES.get(marque, ("", "https://www.google.com"))[1]
    return texte.replace("<url>", f"[page d'aide]({lien})").replace("<num>", "XXXXX")


def afficher_reponse(resultat, numero):
    st.markdown(mettre_en_forme(resultat["reponse"], resultat["marque"]))
    if resultat.get("cas"):
        message_cas, reponse_cas, sim = resultat["cas"]
        with st.expander(f"🔎 Cas similaire retrouvé (similarité {sim:.2f})"):
            st.markdown(f'<div class="cas"><b>Client :</b> {message_cas}<br><br>'
                        f'<b>Agent :</b> {reponse_cas}</div>', unsafe_allow_html=True)
            if sim < 0.2:
                st.caption("⚠️ Cas peu similaire : la réponse s'appuie surtout sur le message.")
    st.markdown('<div class="avis">Cette réponse vous a-t-elle aidé ?</div>', unsafe_allow_html=True)
    avis = st.feedback("thumbs", key=f"avis_{numero}")
    if avis == 1:
        st.caption("Merci pour votre avis ! 🙏")
    elif avis == 0:
        st.caption("Merci, nous allons nous améliorer. 🙏")


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
if "historique" not in st.session_state:
    st.session_state.historique = []

message = st.chat_input("Écrivez votre message au service client…")
message = message or st.session_state.pop("message_exemple", None)

if not st.session_state.historique and not message:
    st.markdown('<div class="vide"><div class="icone">💬</div>'
                'Écrivez un message en anglais, comme un client sur Twitter,<br>'
                'ou essayez un exemple :</div>', unsafe_allow_html=True)
    colonnes = st.columns(2)
    exemples = EXEMPLES.get(marque, []) + EXEMPLES_GENERAUX
    for i, exemple in enumerate(exemples[:4]):
        if colonnes[i % 2].button(exemple, key=f"exemple_{i}", use_container_width=True):
            st.session_state.message_exemple = exemple
            st.rerun()

for numero, tour in enumerate(st.session_state.historique):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(tour["question"])
    with st.chat_message("assistant", avatar="🤖"):
        afficher_reponse(tour, numero)


if message:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(message)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("SupportBot écrit…"):
            resultat = bot.repondre(message, marque, nom_modele, methode, temperature)
        resultat.update(question=message, marque=marque)
        afficher_reponse(resultat, len(st.session_state.historique))
    st.session_state.historique.append(resultat)