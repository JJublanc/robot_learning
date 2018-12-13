from Q_learning import *

ensemble_des_resultats = recuperation_resultats_enregistres()
legend = []
for ii in [2, 3]:
    resultats = ensemble_des_resultats[ii]
    plt.plot(resultats['nombre_mvt'])
    legend.append("alpha = {0} - epsilon = {1} - nb_iter = {2}"
                  .format(resultats['alpha'],
                          resultats['epsilon'],
                          resultats['nb_iter']))
plt.legend(legend, loc='upper right')

