from Q_learning import *

def recuperation_resultats(q=pd.DataFrame(columns=["s", "a", "q"]), nb_iter_prec=0,
                           lamb=0.9, alpha=0.2, epsilon=0.1, nb_iter=1):
    # on lance l'apprentissage
    nombre_mvt, Q_sortie, lamb, alpha, epsilon, nb_iter, type_algo = Q_learning_algo(q, nb_iter_prec, lamb, alpha,
                                                                                     epsilon, nb_iter)

    # on récupère les résultats dans un dictionnaire
    Resultats = {'nombre_mvt': nombre_mvt,
                 'paramètres_apprentissage': Q_sortie,
                 'lamb': lamb,
                 'alpha': alpha,
                 'epsilon': epsilon,
                 'nb_iter': nb_iter,
                 'type_algo': type_algo}
    return Resultats


def enregistrement_resultats(Resultats, mon_fichier=""):
    """
    On récupère le bon numéro pour l'enregistrement du fichier i.e le max utilisé +1
    Puis on enregistre les résultats pickelisés
    :param nom_fichier: nom du fichier de destination
    :param Resultats: résultats d'entrainement
    :return: rien mais enregistre les Resultats dans un fichier
    """
    resultats_numeros = []
    for nom_fichier in os.listdir("resultats"):
        resultats_numeros.append(int(nom_fichier.split("_")[1]))
    try:
        numero_max = max(resultats_numeros)
    except ValueError:
        numero_max = 0

    ## On enregistre les résultat dans un fichier dans le
    if mon_fichier == "":
        chemin_fichier = "resultats\\resultats_{}".format(numero_max + 1)
        with open(chemin_fichier, "wb") as resultats_fichier:
            mon_pickler = pickle.Pickler(resultats_fichier)
            mon_pickler.dump(Resultats)
    else:
        chemin_fichier = "resultats\\{}".format(mon_fichier)
        with open(chemin_fichier, "wb") as resultats_fichier:
            mon_pickler = pickle.Pickler(resultats_fichier)
            mon_pickler.dump(Resultats)