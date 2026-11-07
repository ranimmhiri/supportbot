"""SupportBot : nettoyage, modèles, recherche TF-IDF et génération, pour l'application."""

import html
import math
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, SOS, EOS, SEP = range(5)
TOKENS_INTERDITS = [PAD, UNK, SOS, SEP]


# ---------------------------------------------------------------------------
# 1. Nettoyage (identique au notebook 01_donnees)
# ---------------------------------------------------------------------------
def nettoyer(texte):
    texte = html.unescape(str(texte))
    texte = texte.lower()
    texte = (texte.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'))
    texte = re.sub(r"https?://\S+|www\.\S+", " xxurl ", texte)
    texte = re.sub(r"[@.]?@\w+", " ", texte)
    texte = re.sub(r"(?:\s[-–—]|\s*[\^\*~])\s?[a-z]{1,20}\s*$", " ", texte)
    texte = re.sub(r"\d{5,}", " xxnum ", texte)
    texte = texte.replace("&", " and ").replace("#", " ")
    texte = re.sub(r"[^a-z0-9'.,!?\s]", " ", texte)
    texte = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", texte)
    texte = re.sub(r"([.,!?])\1+", r"\1", texte)
    texte = re.sub(r"(?<!\d)\.|\.(?!\d)", " . ", texte)
    texte = re.sub(r"([,!?])", r" \1 ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    texte = re.sub(r"^(?:[.,!?]\s*)+", "", texte)
    return texte.replace("xxurl", "<url>").replace("xxnum", "<num>")


# ---------------------------------------------------------------------------
# 2. Modèles (identiques au notebook 02_modele)
# ---------------------------------------------------------------------------
class Encodeur(nn.Module):
    def __init__(self, taille_vocab, taille_emb, taille_cachee, dropout):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocab, taille_emb, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(taille_emb, taille_cachee, batch_first=True)

    def forward(self, X):
        longueurs = (X != PAD).sum(dim=1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(X))
        emb = pack_padded_sequence(emb, longueurs, batch_first=True, enforce_sorted=False)
        sorties, (h, c) = self.lstm(emb)
        sorties, _ = pad_packed_sequence(sorties, batch_first=True, total_length=X.size(1))
        return sorties, (h, c)


class Decodeur(nn.Module):
    def __init__(self, taille_vocab, taille_emb, taille_cachee, dropout):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocab, taille_emb, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(taille_emb, taille_cachee, batch_first=True)
        self.sortie = nn.Linear(taille_cachee, taille_vocab)

    def forward(self, Y_entree, etat):
        emb = self.dropout(self.embedding(Y_entree))
        sorties, etat = self.lstm(emb, etat)
        return self.sortie(self.dropout(sorties)), etat


class Seq2Seq(nn.Module):
    def __init__(self, encodeur, decodeur):
        super().__init__()
        self.encodeur = encodeur
        self.decodeur = decodeur


class Attention(nn.Module):
    def __init__(self, taille_cachee):
        super().__init__()
        self.W = nn.Linear(taille_cachee, taille_cachee, bias=False)

    def forward(self, etats_dec, sorties_enc, masque):
        scores = torch.bmm(etats_dec, self.W(sorties_enc).transpose(1, 2))
        scores = scores.masked_fill(~masque.unsqueeze(1), float("-inf"))
        poids = torch.softmax(scores, dim=-1)
        return torch.bmm(poids, sorties_enc), poids


class DecodeurAttention(nn.Module):
    def __init__(self, taille_vocab, taille_emb, taille_cachee, dropout):
        super().__init__()
        self.embedding = nn.Embedding(taille_vocab, taille_emb, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(taille_emb, taille_cachee, batch_first=True)
        self.attention = Attention(taille_cachee)
        self.combinaison = nn.Linear(2 * taille_cachee, taille_cachee)
        self.sortie = nn.Linear(taille_cachee, taille_vocab)

    def forward(self, Y_entree, etat, sorties_enc, masque):
        emb = self.dropout(self.embedding(Y_entree))
        etats_dec, etat = self.lstm(emb, etat)
        contexte, poids = self.attention(etats_dec, sorties_enc, masque)
        combine = torch.tanh(self.combinaison(torch.cat([etats_dec, contexte], dim=-1)))
        return self.sortie(self.dropout(combine)), etat, poids


class Seq2SeqAttention(Seq2Seq):
    pass


def repetitions_interdites(tokens, n=3):
    """Tokens qui recréeraient un groupe de n tokens déjà écrit (blocage des répétitions)."""
    if len(tokens) < n:
        return []
    debut = tuple(tokens[-(n - 1):])                     # les n - 1 derniers tokens
    return [tokens[i + n - 1] for i in range(len(tokens) - n + 1)
            if tuple(tokens[i:i + n - 1]) == debut]


# ---------------------------------------------------------------------------
# 3. Le chatbot : vocabulaire, recherche du cas similaire et génération
# ---------------------------------------------------------------------------
class SupportBot:
    def __init__(self, dossier):
        """Charge le vocabulaire, la mémoire des cas et les trois modèles depuis `dossier`."""
        r = torch.load(f"{dossier}/ressources.pt", map_location="cpu", weights_only=False)
        self.id2mot = r["id2mot"]
        self.mot2id = {mot: i for i, mot in enumerate(self.id2mot)}
        self.idf = r["idf"]
        self.config = r["config"]
        self.memoire = r["memoire"]                       # messages, réponses et marques d'entraînement
        self.M_train = r["M_train"]
        self.marques_train = torch.tensor(
            [r["numeros_marques"][m] for m in self.memoire["marque"]])
        self.numeros_marques = r["numeros_marques"]

        c = self.config
        taille = (len(self.id2mot), c["TAILLE_EMBEDDING"], c["TAILLE_CACHEE"], 0.0)
        self.modeles = {
            "Modèle 1": Seq2Seq(Encodeur(*taille), Decodeur(*taille)),
            "Modèle 2": Seq2SeqAttention(Encodeur(*taille), DecodeurAttention(*taille)),
            "Modèle 3": Seq2SeqAttention(Encodeur(*taille), DecodeurAttention(*taille)),
        }
        for i, modele in enumerate(self.modeles.values(), start=1):
            modele.load_state_dict(torch.load(f"{dossier}/modele{i}.pt", map_location="cpu"))
            modele.eval()

    # --- Représentation -----------------------------------------------------
    def texte_vers_ids(self, texte):
        return [self.mot2id.get(t, UNK) for t in texte.split()]

    def ids_vers_texte(self, ids):
        mots = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, SOS):
                mots.append(self.id2mot[i])
        return " ".join(mots)

    def entree(self, message, cas=None):
        """Numéros d'entrée de l'encodeur : message seul, ou message <SEP> réponse du cas."""
        ids = self.texte_vers_ids(message)[:self.config["MAX_LEN_MESSAGE"]]
        if cas is not None:
            ids += [SEP] + self.texte_vers_ids(cas)[:self.config["MAX_LEN_REPONSE"]]
        return ids

    # --- Recherche du cas similaire (TF-IDF) --------------------------------
    def vecteur_tfidf(self, texte):
        tokens = [t for t in self.texte_vers_ids(texte) if t != UNK]
        poids = {t: n / len(tokens) * self.idf[t] for t, n in Counter(tokens).items()}
        norme = math.sqrt(sum(p ** 2 for p in poids.values()))
        return {t: p / norme for t, p in poids.items()}

    def rechercher(self, message, marque):
        """Message d'entraînement de la même marque le plus similaire : (message, réponse, similarité)."""
        vecteur = torch.zeros(len(self.id2mot))
        for token, poids in self.vecteur_tfidf(message).items():
            vecteur[token] = poids
        sim = torch.sparse.mm(self.M_train, vecteur[:, None])[:, 0]
        sim[self.marques_train != self.numeros_marques.get(marque, -1)] = -1
        i = int(sim.argmax())
        return self.memoire["message"][i], self.memoire["reponse"][i], float(sim[i])

    # --- Génération ----------------------------------------------------------
    def _preparer(self, modele, ids):
        X = torch.tensor([ids])
        sorties_enc, etat = modele.encodeur(X)
        return sorties_enc, etat, X != PAD

    def _etape(self, modele, tokens, etat, sorties_enc, masque):
        """Un pas du décodeur : scores du token suivant, à partir des tokens déjà écrits."""
        entree = torch.tensor([[tokens[-1]]])
        if isinstance(modele, Seq2SeqAttention):
            scores, etat, _ = modele.decodeur(entree, etat, sorties_enc, masque)
        else:
            scores, etat = modele.decodeur(entree, etat)
        scores = scores[0, -1]
        scores[TOKENS_INTERDITS] = float("-inf")
        scores[repetitions_interdites(tokens)] = float("-inf")       # pas deux fois le même trigramme
        return scores, etat

    def generer_greedy(self, modele, ids):
        """Greedy : à chaque étape, le token le plus probable."""
        return self.generer_echantillonnage(modele, ids, temperature=0)

    def generer_echantillonnage(self, modele, ids, temperature=0.8, top_k=20):
        """Échantillonnage : on tire le token suivant au hasard parmi les `top_k` plus probables.
        Plus la température est haute, plus le choix est varié ; à 0, c'est le greedy."""
        with torch.no_grad():
            sorties_enc, etat, masque = self._preparer(modele, ids)
            tokens = [SOS]
            for _ in range(self.config["MAX_LEN_REPONSE"]):
                scores, etat = self._etape(modele, tokens, etat, sorties_enc, masque)
                if temperature == 0:
                    token = scores.argmax().item()
                else:
                    valeurs, indices = (scores / temperature).topk(top_k)
                    probas = torch.softmax(valeurs, dim=-1)
                    token = indices[torch.multinomial(probas, 1)].item()
                if token == EOS:
                    break
                tokens.append(token)
        return tokens[1:]

    def generer_beam(self, modele, ids, largeur=5):
        """Beam search : on garde les `largeur` meilleurs débuts de phrase à chaque étape."""
        with torch.no_grad():
            sorties_enc, etat, masque = self._preparer(modele, ids)
            faisceaux, terminees = [([SOS], 0.0, etat)], []
            for _ in range(self.config["MAX_LEN_REPONSE"]):
                candidats = []
                for tokens, score, etat in faisceaux:
                    scores, nouvel_etat = self._etape(modele, tokens, etat, sorties_enc, masque)
                    valeurs, indices = torch.log_softmax(scores, dim=-1).topk(largeur)
                    for log_p, token in zip(valeurs.tolist(), indices.tolist()):
                        candidats.append((tokens + [token], score + log_p, nouvel_etat))
                candidats = sorted(candidats, key=lambda c: c[1], reverse=True)[:largeur]
                faisceaux = []
                for candidat in candidats:
                    if candidat[0][-1] == EOS:
                        terminees.append(candidat)
                    else:
                        faisceaux.append(candidat)
                if not faisceaux:
                    break
            meilleure = max(terminees + faisceaux, key=lambda c: c[1] / len(c[0]))
        return [t for t in meilleure[0][1:] if t != EOS]

    def attention(self, modele, ids_entree, ids_reponse):
        """Poids d'attention du modèle pour une réponse donnée : (tokens de la réponse, tokens d'entrée)."""
        with torch.no_grad():
            sorties_enc, etat, masque = self._preparer(modele, ids_entree)
            Y = torch.tensor([[SOS] + ids_reponse])
            _, _, poids = modele.decodeur(Y, etat, sorties_enc, masque)
        return poids[0, :len(ids_reponse)].numpy()           # une ligne par token de la réponse

    # --- Tout le chemin : message -> réponse --------------------------------
    def repondre(self, message, marque, nom_modele="Modèle 3", methode="beam", temperature=0.8):
        message = nettoyer(message)
        modele = self.modeles[nom_modele]
        cas = self.rechercher(message, marque) if nom_modele == "Modèle 3" else None
        ids = self.entree(message, cas[1] if cas else None)
        if methode == "beam":
            ids_reponse = self.generer_beam(modele, ids)
        elif methode == "creatif":
            ids_reponse = self.generer_echantillonnage(modele, ids, temperature)
        else:
            ids_reponse = self.generer_greedy(modele, ids)
        resultat = {"message": message, "reponse": self.ids_vers_texte(ids_reponse), "cas": cas}
        if isinstance(modele, Seq2SeqAttention) and ids_reponse:
            resultat["attention"] = self.attention(modele, ids, ids_reponse)
            resultat["tokens_entree"] = [self.id2mot[i] for i in ids]
            resultat["tokens_reponse"] = [self.id2mot[i] for i in ids_reponse]
        return resultat