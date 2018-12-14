from resultats_enregistrement import *

def recuperation_resultats_enregistres():
    """
    :return: ensemble des résultat enregistrés dans le dossier resultats, sous forme de liste
    """
    ensemble_des_resultats = list()
    for fichier in os.listdir('resultats'):
        path_fichier_a_ouvrir = 'resultats\\{}'.format(fichier)
        with open(path_fichier_a_ouvrir, 'rb') as fichier_lu:
            mes_resultats = pickle.load(fichier_lu)
            ensemble_des_resultats.append(mes_resultats)
    return ensemble_des_resultats

def afficher_graph(liste_numeros_resultats=[2, 3]):
    """
    :param liste_numeros_resultats: liste de numéro de fichier que l'on souhaite afficher
    :return: vide
    : action : créer un graphique
    """
    legend = []
    for ii in liste_numeros_resultats:
        resultats = ensemble_des_resultats[ii]
        plt.plot(resultats['nombre_mvt'])
        legend.append("alpha = {0} - epsilon = {1} - nb_iter = {2}"
                      .format(resultats['alpha'],
                              resultats['epsilon'],
                              resultats['nb_iter']))
    plt.legend(legend, loc='upper right')


ensemble_des_resultats = recuperation_resultats_enregistres()
afficher_graph([2, 3])
