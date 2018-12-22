from resultats_enregistrement import *
import pandas as pd


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


def recuperation_resultat_par_nom(noms_fichiers):
    """
    :return: ensemble des résultat enregistrés dans le dossier resultats, sous forme de liste
    """
    liste_des_resultats = list()
    for fichier in noms_fichiers:
        path_fichier_a_ouvrir = 'resultats\\{}'.format(fichier)
        with open(path_fichier_a_ouvrir, 'rb') as fichier_lu:
            mes_resultats = pickle.load(fichier_lu)
            liste_des_resultats.append(mes_resultats)
    return liste_des_resultats


def afficher_graph(liste_numeros_resultats=[2, 3]):
    """
    :param liste_numeros_resultats: liste de numéro de fichier que l'on souhaite afficher
    :return: vide
    : action : créer un graphique
    """
    ensemble_des_resultats = recuperation_resultats_enregistres()
    legend = []
    for ii in liste_numeros_resultats:
        resultats = ensemble_des_resultats[ii]
        plt.plot(resultats['nombre_mvt'])
        legend.append("alpha = {0} - epsilon = {1} - nb_iter = {2}"
                      .format(resultats['alpha'],
                              resultats['epsilon'],
                              resultats['nb_iter']))
    plt.legend(legend, loc='upper right')


# sorties

# afficher_graph([2, 3])
def resultats_premiers_pas():
    resultats_1_2_iterations = list()
    for fichier in ['resultats_1_itérations', 'resultats_2_itérations']:
        path_fichier_a_ouvrir = 'resultats\\{}'.format(fichier)
        with open(path_fichier_a_ouvrir, 'rb') as fichier_lu:
            mes_resultats = pickle.load(fichier_lu)
            resultats_1_2_iterations.append(mes_resultats)
    return resultats_1_2_iterations


def afficher_position_valeurQ(resultats_observes):
    for resultat in resultats_observes:
        result = resultat['paramètres_apprentissage']
        result_tri = result.sort_values('q', ascending=False).reset_index()
        result_tri = result_tri.loc[result_tri.q != 0, :]
        for ii in range(len(result_tri)):
            q = result_tri.loc[ii, 'q']
            nb_iter = resultat['nb_iter']
            lamb = resultat['lamb']
            alpha = resultat['alpha']
            a = result_tri.loc[ii, 'a']
            print(result_tri.loc[ii, 's'])
            print("Après {} itération et avec un discount de {}\n"
                  "et un taux d'apprentissage de {},\n"
                  "dans cette état s la valeur de l'action {} est {}".format(nb_iter, lamb, alpha, a, q))
            input()


resultats_observes = resultats_premiers_pas()
afficher_position_valeurQ(resultats_observes)
afficher_graph()

#########################
## graphique principal ##
#########################
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# renvoie la légende du graphique
def create_legende(resultats):
    res_series = pd.Series(resultats['nombre_mvt'])
    nb_iter = resultats['nb_iter']
    moyenne_derniers_coups = round(res_series[(nb_iter - 10): nb_iter].mean())
    moyenne_premiers_coups = round(res_series[0:10].mean())
    legende = "alpha = {0} - epsilon = {1} - nb_iter = {2}\n" \
              "Nb de mvts moyen 10 derniers essais : {3}\n" \
              "Nb de mvts moyen 10 premiers essais : {4}".format(resultats['alpha'],
                                                                                   resultats['epsilon'],
                                                                                   nb_iter,
                                                                                   moyenne_derniers_coups,
                                                                                   moyenne_premiers_coups)
    return legende


# récupération des résultats
resultats_affiches = recuperation_resultat_par_nom(["resultats_7", "resultats_8"])

# création du graphique subplots
fig, axes = plt.subplots(1, 2)

res_1 = resultats_affiches[0]
res_2 = resultats_affiches[1]
df1 = pd.Series(res_1['nombre_mvt'])
df2 = pd.Series(res_2['nombre_mvt'])
y_max = max(df1.max(), df2.max())
x_max = max(len(df1), len(df2))
loc = plticker.MultipleLocator(base=15)

ax_1 = df1.plot(ax=axes[0], label=create_legende(res_1), color="#ff1a1a")
ax_1.legend(loc="upper right", prop={'size': 12})
ax_1.set_xlabel("numéro de l'essai", fontSize=18)
ax_1.set_ylabel("nombre de mouvements par essai", fontSize=18)
ax_1.set_ylim([0, y_max])
ax_1.set_xlim([0, x_max])
ax_1.xaxis.set_major_locator(loc)

ax_2 = df2.plot(ax=axes[1], label=create_legende(res_2), color="#0052cc")
ax_2.legend(loc="upper right", prop={'size': 12})
ax_2.set_xlabel("numéro de l'essai", fontSize=18)
ax_2.set_ylim([0, y_max])
ax_2.set_xlim([0, x_max])
ax_2.xaxis.set_major_locator(loc)

if res_1['type_algo'] == res_2['type_algo']:
    fig.suptitle("Résultats d'un entrainement de \"robot\" avec un algorithme de type {}".format(res_1['type_algo']),
                 fontsize=25)