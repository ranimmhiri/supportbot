"""SupportBot : l'application de démonstration du chatbot (streamlit run app/app.py)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
[data-testid="stChatMessage"] { border-radius: 16px; padding: 0.9rem 1rem; margin-bottom: 0.6rem; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: #eef4fc; border: 1px solid #d6e4f7;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
    background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 1px 6px rgba(15, 23, 42, 0.05);
}
.resultats { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.resultats th { text-align: left; color: #6b7280; font-weight: 600; padding: 0.3rem 0.2rem; }
.resultats td { padding: 0.35rem 0.2rem; border-top: 1px solid #e5e7eb; }
.resultats tr.meilleur td { color: #2f9e74; font-weight: 700; }
.titre, .titre p {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0; line-height: 1.2;
    background: linear-gradient(90deg, #2a78d6 0%, #2f9e74 55%, #eb6834 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sous-titre { color: #6b7280; font-size: 1.02rem; margin-top: 0.2rem; margin-bottom: 1.4rem; }
.badge {
    display: inline-block; padding: 0.18rem 0.7rem; margin: 0 0.35rem 0.35rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: #eef4fc; color: #2a78d6; border: 1px solid #d6e4f7;
}
.cas {
    border-left: 3px solid #2f9e74; background: #f3faf7; border-radius: 8px;
    padding: 0.7rem 0.9rem; font-size: 0.9rem; color: #374151; margin-top: 0.3rem;
}
.cas b { color: #2f9e74; }
.vide {
    text-align: center; color: #6b7280; padding: 2.2rem 1rem 1rem 1rem;
    border: 1px dashed #d1d5db; border-radius: 16px; margin-bottom: 1rem;
}
.vide .icone { font-size: 2.4rem; }
[data-testid="stSidebar"] { background: #f8fafc; }
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
    voir_attention = st.toggle("Afficher la carte d'attention", value=False,
                               disabled=nom_modele == "Modèle 1")
    st.divider()
    st.markdown("### 📊 Résultats sur le test")
    st.markdown("""
<table class="resultats">
<tr><th></th><th>Perplexité</th><th>BLEU</th></tr>
<tr><td>Modèle 1</td><td>12,72</td><td>6,47</td></tr>
<tr><td>Modèle 2</td><td>11,47</td><td>7,16</td></tr>
<tr class="meilleur"><td>Modèle 3</td><td>10,71</td><td>9,99</td></tr>
</table>
""", unsafe_allow_html=True)
    st.caption("Encodeur–décodeur LSTM construit et entraîné from scratch (PyTorch), "
               "sur 150 000 conversations de service client sur Twitter.")
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
st.markdown(f'<span class="badge">🏢 {MARQUES[marque][0]}</span>'
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


def carte_attention(resultat):
    """Carte des poids d'attention : tokens de la réponse en lignes, tokens d'entrée en colonnes."""
    entree, reponse = resultat["tokens_entree"], resultat["tokens_reponse"]
    fig, ax = plt.subplots(figsize=(max(7, 0.3 * len(entree)), min(8, max(3, 0.3 * len(reponse)))))
    sns.heatmap(resultat["attention"], xticklabels=entree, yticklabels=reponse, cmap="YlOrRd",
                linewidths=0.2, cbar_kws={"label": "Poids d'attention", "shrink": 0.7}, ax=ax)
    ax.set_xlabel("Entrée de l'encodeur" + (" (message <SEP> réponse du cas)" if "<SEP>" in entree else ""))
    ax.set_ylabel("Réponse générée")
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    return fig


def afficher_reponse(resultat):
    st.markdown(mettre_en_forme(resultat["reponse"], resultat["marque"]))
    if resultat.get("cas"):
        message_cas, reponse_cas, sim = resultat["cas"]
        with st.expander(f"🔎 Cas similaire retrouvé (similarité {sim:.2f})"):
            st.markdown(f'<div class="cas"><b>Client :</b> {message_cas}<br><br>'
                        f'<b>Agent :</b> {reponse_cas}</div>', unsafe_allow_html=True)
            if sim < 0.2:
                st.caption("⚠️ Cas peu similaire : la réponse s'appuie surtout sur le message.")
    if resultat.get("voir_attention") and "attention" in resultat:
        with st.expander("👀 Où le modèle a-t-il regardé ?"):
            st.pyplot(carte_attention(resultat))


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

for tour in st.session_state.historique:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(tour["question"])
    with st.chat_message("assistant", avatar="🤖"):
        afficher_reponse(tour)


if message:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(message)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("SupportBot écrit…"):
            resultat = bot.repondre(message, marque, nom_modele, methode, temperature)
        resultat.update(question=message, marque=marque, voir_attention=voir_attention)
        afficher_reponse(resultat)
    st.session_state.historique.append(resultat)