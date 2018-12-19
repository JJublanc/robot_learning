from resultats_enregistrement import *


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
# entrainement_modeles(nb_iter=300)
# TODO ajouter un paramètre correspondant au temps d'entrainement du modèle

# On entraine deux modèles pour avoir des données après 1 et 2 itérations et faire des vérifications
# pour la présentation
for ii in [1, 2]:
    nom_mon_fichier = "resultats_{}_itérations".format(ii)
    enregistrement_resultats(recuperation_resultats(nb_iter=ii, alpha=0.5, epsilon=0.2), mon_fichier=nom_mon_fichier)
