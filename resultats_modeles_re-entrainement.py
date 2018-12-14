from Q_learning import *

# On ré-entraine un modèle
# Et on l'enregistre au même endroit
nb_fichier = 0
ensemble_des_resultats = recuperation_resultats_enregistres()
q = ensemble_des_resultats[nb_fichier]['paramètres_apprentissage']
alpha = ensemble_des_resultats[nb_fichier]['alpha']
epsilon = ensemble_des_resultats[nb_fichier]['epsilon']
nb_iter_prec = ensemble_des_resultats[nb_fichier]['nb_iter']
resultats = recuperation_resultats(q, nb_iter_prec=nb_iter_prec, nb_iter=10, alpha=alpha, epsilon=epsilon)
nom_fichier = os.listdir('resultats')[nb_fichier]
chemin_fichier = "resultats\\{}".format(nom_fichier)
with open(chemin_fichier, "wb") as resultats_fichier:
    mon_pickler = pickle.Pickler(resultats_fichier)
    mon_pickler.dump(resultats)

#TODO a finaliser