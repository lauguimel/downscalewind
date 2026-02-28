# services/module3-stochastic

**Statut : OPTIONNEL — V2 uniquement**

## Responsabilité

Modélisation de la turbulence sous-horaire (fluctuations < 1h non captées
par le surrogate RANS stationnaire).

## Motivation

Les sorties du Module 2B (GNN surrogate) représentent un champ moyen stationnaire.
Pour les applications outdoor (parapente, planeur) qui nécessitent des informations
sur la variabilité rapide du vent, il faut simuler les fluctuations turbulentes
sur des échelles de temps de quelques minutes.

## Approches potentielles (à évaluer)

1. **Méthode stochastique de Mann (1998)** : génération de champs de turbulence
   3D cohérents spectralement, calibrés sur les spectres de von Kármán. Standard
   IEC 61400-1 pour la certification éolienne.

2. **SDE conditionnelle** : résoudre dX = f(X,t)dt + σ(X,t)dW avec f issu du
   surrogate RANS et σ calibré sur les données Perdigão.

3. **Diffusion conditionnelle** : modèle de score-matching entraîné sur les
   fluctuations haute-fréquence des mâts Perdigão.

## Décision

Ce module n'est pas implémenté en V1. Le PoC utilise uniquement les champs
moyens du Module 2B. L'ajout de turbulence stochastique est prévu en V2
après validation des modules 1 et 2.

## Questions ouvertes

- Quelle méthode stochastique est la plus adaptée à la contrainte de latence
  < 500 ms pour l'application temps réel ?
- Les spectres turbulents de Perdigão sont-ils représentatifs d'autres sites ?
- Comment conditionner le modèle stochastique sur la stabilité atmosphérique ?
