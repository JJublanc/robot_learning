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
                    lamb=0.9, alpha=0.8, epsilon=0.1, nb_iter=1,
                    verbosity=True):
    """
    q = paramètres d'apprentissages, par défaut on repart de 0, mais on peut repartir d'un ancien modèle
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
        """
        Initialisation : chargement du labyrinthe
        """
        cartes = chargement_des_cartes()
        labyrinthe = cartes[0].labyrinthe
        labyrinthe.messages = False  # on ne souhaite pas avoir de messages dans la console
        placement_robot = random.choice(labyrinthe.places_libres)  # on choisit un placement du robot au hasard
        labyrinthe.changer_position(placement_robot[0], placement_robot[1])  # on déplace le robot
        fin_partie = False
        nbnb_essais = 0

        """
        Boucle pour une partie
        """
        while not fin_partie:
            # état de l'environnement : ici c'est le labyrinthe (position des obstacles, portes, sorties et robot)
            s = '\n'.join(["".join(line) for line in labyrinthe.grille])

            # on choisit une action avec la politique epsilon-greedy et en fonction de :
            # * l'état de l'environnement s
            # * la connaissance Q
            a = action_choisie(epsilon, s, Q)

            # on effectue l'action choisie, ici on bouge (ou plutôt on tente de bouger) le robot
            # NB : la fonction effectue une transformation de l'état du labyrinthe
            # Et renvoit un booleen indiquant si la partie est finie ou non
            fin_partie = labyrinthe.executer_instruction(a)

            # on récupère la récompense imédiate
            r = labyrinthe.nombre_points

            # on récupère le nouvel état de l'environnement
            s_prim = '\n'.join(["".join(line) for line in labyrinthe.grille])

            # On calcule la target qui est égale à :
            #  - la récompense immédiate
            #  - à laquelle s'ajoute la valeur q maximale attendue dans le nouvel état s_prim
            # Si on ne peut pas calculer l'espérance de gain on l'estime à 0
            target = def_target(s_prim, Q, lamb, r)

            """
            On met à jour la valeur q pour le couple s,a (état-action)
            Il y a deux cas : 
            * dans le premier cas on a déjà une valeur pour le couple (s,a) et on met à jour la valeur
            * dans l'autre cas on initialise avec comme valeur celle de la target
            """

            if sum(Q[(Q.s == s)].loc[:, 'a'].isin([a])) > 0:
                Q.loc[(Q.s == s) & (Q.a == a), 'q'] = (alpha * Q[(Q.s == s) & (Q.a == a)].q
                                                       + (1 - alpha) * target)
                if verbosity:
                    print("mise à jour {}".format(Q.loc[(Q.s == s) & (Q.a == a), 'q']))
            else:
                df = pd.DataFrame([[s, a, target]], columns=["s", "a", "q"])
                if verbosity:
                    print("ajout {}".format(df.s))
                Q = Q.append(df)
            nbnb_essais += 1
        nombre_mvt.append(nbnb_essais)
    return nombre_mvt, Q, lamb, alpha, epsilon, nb_iter


def def_target(s_prim, Q, lamb, r):
    """
    Calcul de la valeur cible à l'instant t
    :param s_prim: état suivant l'action
    :param Q: connaissance sur l'environnement
    :param lamb: facteur de discount
    :param r: récompense suite à l'action
    :return: valeur des gains totaux esperée après avoir reçu la récompense r et être arrivé dans l'état s_prim
    """
    if s_prim in list(Q.s):
        target = r + lamb * max(Q[Q.s == s_prim].q)  # récompense immédiates + récompenses de long termes espérées
    else:
        target = r
    return target


def action_choisie(epsilon, s, Q):
    """
    choix d'une action avec une politique epsilon-greedy
    :param epsilon: taux de dérogation à une politique purement greedy
    :param s: état de l'environnement
    :param Q: connaissance sur l'environnement
    :return: action choisie
    """
    rand = np.random.choice((0, 1), p=[(1 - epsilon), epsilon])
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
    return a


def recuperation_resultats(q=pd.DataFrame(columns=["s", "a", "q"]),
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


def enregistrement_resultats(Resultats):
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


def recuperation_resultats_enregistres():
    ensemble_des_resultats = list()
    for fichier in os.listdir('resultats'):
        path_fichier_a_ouvrir = 'resultats\\{}'.format(fichier)
        with open(path_fichier_a_ouvrir, 'rb') as fichier_lu:
            mes_resultats = pickle.load(fichier_lu)
            ensemble_des_resultats.append(mes_resultats)
    return ensemble_des_resultats
