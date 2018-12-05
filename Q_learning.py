# -*- coding: utf-8 -*-
import random
from actions import *
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time


def Q_learning_algo(q=pd.DataFrame(columns=["s", "a", "q"]),
                    lamb=0.9, alpha=0.8, epsilon=0.1, nb_iter=1):
    """
    q = paramètres d'apprentissages par défaut on repart de 0 mais on peut repartir d'un ancien modèle
    lamb = 0.9  # facteur de discount
    alpha = 0.8  # taux d'apprentissage
    epsilon = 0.1  # paramètre epsilon pour la epsilon-greedy policy
    """
    try:
        Q = q
        assert isinstance(Q, pd.DataFrame)
        assert [q.columns] == ["s", "a", "q"]
    except AssertionError:
        print("Q doit être un dataFrame avec trois colonnes nommées s, a et q")
        Q = pd.DataFrame(columns=["s", "a", "q"])

    nombre_mvt = []

    """
    Boucle pour plusieurs parties
    """
    for ii in range(nb_iter):
        cartes = chargement_des_cartes()
        labyrinthe = cartes[0].labyrinthe
        placement_robot = random.choice(labyrinthe.places_libres)
        labyrinthe.changer_position(placement_robot[0], placement_robot[1])
        fin_partie = False
        nbnb_essais = 0

        """
        Boucle pour une partie
        """
        while not fin_partie:
            # Etat de l'environnement : ici c'est le labyrinthe (position des obstacles, portes, sorties et robot)
            s = '\n'.join(["".join(line) for line in labyrinthe.grille])

            rand = np.random.choice((0, 1), p=[epsilon, (1 - epsilon)])
            # si l'état est connu et qu'on est dans une configuration greedy on choisi
            # l'action maximisant l'espérance de gain
            # sinon on effectue un mouvement au hasard
            if (s in list(Q.s)) & (rand == 1):
                if max(Q[Q.s == s].q) > 0:
                    max_q = max(Q[Q.s == s].q)
                    print(max_q)
                    a = random.choice(Q[(Q.s == s) & (Q.q == max_q)].a)
                else:
                    a = random.choice(['n', 's', 'e', 'o'])
            else:
                a = random.choice(['n', 's', 'e', 'o'])

            # on effectue l'action choisie, ici on bouge (on tente de bouger) le robot
            # afficher_labyrinthe(labyrinthe)
            fin_partie = labyrinthe.executer_instruction(a)
            # afficher_labyrinthe(labyrinthe)

            if fin_partie:
                print("stop")

            # on récupère la récompense imédiate
            r = labyrinthe.nombre_points
            # on récupère le nouvel état de l'environnement
            s_prim = '\n'.join(["".join(line) for line in labyrinthe.grille])

            # On calcule la target qui est égale à :
            #  - la récompense immédiate
            #  - à laquelle s'ajoute la valeur q maximale attendue dans le nouvel état s_prim
            # Si on ne peut pas calculer l'espérance de gain on l'estime à 0
            if s_prim in list(Q.s):
                target = r + lamb * max(Q[Q.s == s_prim].q)
            else:
                target = r

            # On met à jour la valeur q pour le couple s,a (état-action)
            if sum(Q[(Q.s == s)].loc[:, 'a'].isin([a])) > 0:
                Q.loc[(Q.s == s) & (Q.a == a), 'q'] = (alpha * Q[(Q.s == s) & (Q.a == a)].q
                                                       + (1 - alpha) * target)
                print("mise à jour {}".format(Q.loc[(Q.s == s) & (Q.a == a), 'q']))
            else:
                df = pd.DataFrame([[s, a, target]], columns=["s", "a", "q"])
                print("ajout {}".format(df.s))
                Q = Q.append(df)
            nbnb_essais += 1
        nombre_mvt.append(nbnb_essais)
    return nombre_mvt, Q, lamb, alpha, epsilon, nb_iter


def Q_learning_resultats(q=pd.DataFrame(columns=["s", "a", "q"]),
                         lamb=0.9, alpha=0.8, epsilon=0.1, nb_iter=1):
    nombre_mvt, Q_sortie, lamb, alpha, epsilon, nb_iter = Q_learning_algo(q, lamb, alpha, epsilon, nb_iter)
    # pd.Series(resultats).plot()
    Resultats = {'nombre_mvt': nombre_mvt,
                 'paramètres_apprentissage': Q_sortie,
                 'lamb': lamb,
                 'alpha': alpha,
                 'epsilon': epsilon,
                 'nb_iter': nb_iter}
    return Resultats


def Q_learning_enregistrement(Resultats):
    """
    On récupère le bon numéro pour l'enregistrement du fichier i.e le max utilisé +1
    Puis on enregistre les résultats pickelisés
    :param Resultats:
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
    nom_fichier = "resultats\\resultats_{}".format(numero_max + 1)
    with open(nom_fichier, "wb") as resultats_fichier:
        mon_pickler = pickle.Pickler(resultats_fichier)
        mon_pickler.dump(Resultats)


# TODO empecher d'avoir des labyrinthe avec des lignes de tailles différentes ou des colonnes de tailles differentes

def entrainement_modeles():
    for alpha in [0.5, 0.8]:
        for epsilon in [0.7, 0.9]:
            Q_learning_enregistrement(Q_learning_resultats(nb_iter=100, alpha=alpha, epsilon=epsilon))


def recuperation_resultats_enregistres():
    ensemble_des_resultats = list()
    for fichier in os.listdir('resultats'):
        path_fichier_a_ouvrir = 'resultats\\{}'.format(fichier)
        with open(path_fichier_a_ouvrir, 'rb') as fichier_lu:
            mes_resultats = pickle.load(fichier_lu)
            ensemble_des_resultats.append(mes_resultats)
    return ensemble_des_resultats


ensemble_des_resultats = recuperation_resultats_enregistres()
legend = []
for ii in [2,3]:
    resultats = ensemble_des_resultats[ii]
    plt.plot(resultats['nombre_essai'])
    legend.append("alpha = {0} - epsilon = {1} - nb_iter = {2}"
                  .format(resultats['alpha'],
                          resultats['epsilon'],
                          resultats['nb_iter']))
plt.legend(legend, loc='upper right')

import tkinter

cartes = chargement_des_cartes()
labyrinthe = cartes[0].labyrinthe
placement_robot = random.choice(labyrinthe.places_libres)
labyrinthe.changer_position(placement_robot[0], placement_robot[1])
fin_partie = False

