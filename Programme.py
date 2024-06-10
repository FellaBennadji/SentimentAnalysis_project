import pathlib
from gensim.models import KeyedVectors as kv
import spacy
from scipy.stats import hmean
import json
import sys

# chemin vers le fichier des plongement lexicaux
embfile = "./frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"

# Charger les plongements lexicaux en mémoire
wv = kv.load_word2vec_format(embfile, binary=True, encoding='UTF-8', unicode_errors='ignore')

# Charger spacy avec le modèle du français
spacy_nlp = spacy.load('fr_core_news_md')


# Pour chacun des trois aspects, on fournit des mots-exemples qui seront utilisés pour calculer
# des scores de similarité avec chaque token du texte afin de décider s'il exprime un des
# trois aspects
aspects = {
    'nourriture': ['dessert', 'poisson', 'riz', 'pâtes', 'purée', 'viande', 'sandwich', 'frites'],
    'boisson': ['eau', 'vin', 'limonade', 'bière', 'jus', 'thé', 'café'],
    'service': ["service", 'serveur', 'patron', 'employé'],
}


# Similarité moyenne entre un mot et un ensemble de mots : on prend la moyenne harmonique des distances
# puis on la soustrait à 1 pour obtenir la mesure inverse (la similarité), et on arrondit à 4 décimales.
def get_sim(word, other_words):
    if word not in wv.key_to_index:
        return 0
    dpos = wv.distances(word, other_words)
    d = hmean(abs(dpos))
    return round((1 - d),4)


# Pour un token spacy, cette méthode décide si c'est un terme d'aspect en cherchant l'aspect pour
# lequel il a une similarité maximale (calculée avec les mots-exemples des aspetcs).
# si le score maxi est plus petit que 0.5, il n'y pas d'aspect et la méthode retourne None
def get_aspect_emb(token):
    if token.pos_ != "NOUN":
        return None
    aspect_names = [aspect_name for aspect_name in aspects]
    scores = [(aspect_name,get_sim(token.lemma_, aspects[aspect_name])) for aspect_name in aspect_names]
    scores.sort(key=lambda x:-x[1])
    max_score = scores[0][1]
    max_aspect = scores[0][0] if max_score >= 0.5 else None
    return max_aspect


######################################################################################################
with open(sys.argv[1], 'r', encoding = 'UTF-8') as tf: ## chargement des textes dans une liste
    textes = tf.read().split('\n')

docs = spacy_nlp.pipe(textes) ## analyse des textes avec spaCy

resultats = []
for doc in docs:
    for sent in doc.sents:
        triplets = []
        Aspect = []
        lastaspect = ''
        Term = ''
        term = []
        carac = []
        resultat = []
        for token in sent:
            #print(token.text, token.dep_, token.head.text, token.head.pos_,
                  #token.pos_, [child for child in token.children])
            aspect = get_aspect_emb(token)
            if aspect is not None:
                Aspect.append(aspect)
                term.append(token.text)
            c = token.text
            if (token.pos_ == 'ADJ' and token.dep_ == "amod" and token.head.text in term) or (
                    token.pos_ == 'ADJ' and token.head.pos_ == "ADJ") or (
                    token.pos_ == 'ADJ' and token.dep_ == 'conj'):
                c = token.text
                indicateur = False
                for child in token.children:
                    if child.pos_ == 'ADV' and child.lemma_ in ["pas", "non", "peu", "guère", "mal", "trop"]:
                        c = child.text + '_' + token.text
                        indicateur = True
                    if not indicateur:
                        c = token.text
                carac.append(c)
        if Aspect and carac:
            lastaspect = Aspect[0]
            Term = term[0]
            for adj in carac :
                triplets.append((lastaspect, Term, adj))
        resultat = {'phrase': sent.text, 'triplets': triplets} ## sauvegarde du r´esultat pour cette phrase
        resultats.append(resultat)

resultats_json = json.dumps(resultats, indent=4, ensure_ascii=False)
resultats_json_file = open("resultats.json", "w", encoding='UTF-8')
resultats_json_file.write(resultats_json)
resultats_json_file.close()
