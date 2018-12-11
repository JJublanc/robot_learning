from actions import *
import pytest


def test_charger_carte():
    """
    on teste le chargement des cartes
    """
    assert type(chargement_des_cartes()) == list


def test_instanciation_labyrinthe():
    """
    on teste la création d'un labyrinthe à partir d'une carte
    """
    assert isinstance(chargement_des_cartes()[0].labyrinthe, Labyrinthe)


@pytest.fixture()
def carte_principale():
    path_main_map = os.path.join("cartes_tests", "carte_principale.txt")
    with open(path_main_map, "r") as fichier:
        carte_txt = fichier.read()
        carte_principale = Carte("carte_principale", carte_txt)
    return carte_principale


@pytest.fixture()
def carte_principale_ouest():
    path_main_map = os.path.join("cartes_tests", "carte_principale_ouest.txt")
    with open(path_main_map, "r") as fichier:
        carte_txt = fichier.read()
        carte_principale_ouest = Carte("carte_principale_ouest", carte_txt)
    return carte_principale_ouest


def test_mouvement():
    """
    on compare la grille du labyrinthe après un mouvement vers l'ouest
    avec une grille contenant le résultat attendu
    :return:
    """
    # on importe le labyrinthe principal
    labyrinthe_principal = carte_principale().labyrinthe
    # on effectue un mouvement vers l'ouest à partir du labyrinthe principal
    labyrinthe_principal.executer_instruction("o")
    # on importe le labyrinthe attendu
    labyrinthe_principal_ouest = carte_principale_ouest().labyrinthe
    assert labyrinthe_principal.grille == labyrinthe_principal_ouest.grille