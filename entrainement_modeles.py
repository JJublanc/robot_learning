from Q_learning import *


def entrainement_modeles(alphas=[0.2, 0.5], epsilons=[0.3, 0.1], nb_iter=1):
    """
    entraine les modèles avec les combinaisons de paramètres enregistrées
    recupère les résultats de tous les modèles
    enregistre les résultat dans un fichier
    :param alphas: liste des valeurs pour le parametre epsilon
    :param epsilons: liste des valeurs pour le parametre epsilon
    :param nb_iter: nombre d'iterations
    :return: vide
    """
    for alpha in alphas:
        for epsilon in epsilons:
            enregistrement_resultats(recuperation_resultats(nb_iter=nb_iter, alpha=alpha, epsilon=epsilon))


# On entraine un premier jeu de données
entrainement_modeles(nb_iter=300)

# On ré-entraine un modèle
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