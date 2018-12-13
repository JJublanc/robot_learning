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


entrainement_modeles()
