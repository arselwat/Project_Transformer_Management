# Pages corrigées — fiabilité et maintenance des transformateurs

## Installation

1. Arrêter Streamlit et conserver une copie du projet actuel.
2. Extraire le contenu de cette archive dans le dossier qui contient votre `app.py`.
3. Accepter le remplacement des cinq fichiers de `pages/` et des trois modules de calcul. Le fichier `core/workflow.py` est nouveau et indispensable.
4. Si nécessaire, installer les dépendances : `python -m pip install -r requirements_pages.txt`.
5. Relancer depuis la racine : `python -m streamlit run app.py`.
6. Se connecter avec le mécanisme existant, puis réimporter le fichier d'exploitation dans Sources de données.

Ne pas renommer les cinq pages : la navigation existante utilise leurs chemins.
Cette archive est une mise à jour du projet déjà fourni, pas une application autonome.
Elle utilise `core/ui.py`, `core/security/auth.py` et `app.py` de votre projet.
Elle ne contient aucun identifiant de connexion ni aucune donnée d'exploitation.

## Contenu

- `core/datahub.py` : données, dates, fenêtres et contrôles.
- `core/reliability/organigram.py` : comparaison de modèles, validation et prévision conditionnelle.
- `core/reliability/optimize.py` : horizons et coût prospectif PLP/HPP.
- `core/workflow.py` : partage des résultats, invalidation et exports.
- `pages/1_Sources_fully_linked_fixed.py` : importation et période observée.
- `pages/2_Indicateurs_verified.py` : réglages, estimation, diagnostics, validation et graphiques.
- `pages/3_Optimisation_verified.py` : cibles, coûts et contraintes.
- `pages/4_Maintenance_verified.py` : échéance et proposition motivée par l'exploitant.
- `pages/5_Resultat_analyse_optimisation_Maintenance_fixed.py` : synthèse et exports.

Les pages Stock, Transformateurs, Temps réel et Alertes, ainsi que l'accueil et le mécanisme de connexion, restent dans le projet existant. Elles ne font pas partie de cette livraison. Aucun appel à `unify.py` n'est utilisé dans le nouveau parcours des cinq pages.

## Préparer les données

CSV d'événements : `asset_id`, `event_start`, `is_failure` obligatoires.
Les alias `equipment_code` et `timestamp` sont acceptés pour ce format.
Les colonnes `event_id`, `event_end`, `is_planned`, `repair_time_hours` et `downtime_hours` sont optionnelles. `event_end` représente la remise en service. Un indicateur présent mais ambigu bloque l'analyse de l'équipement concerné.

Utiliser les dates ISO : `2025-11-09 08:40:00`, dans une même base de temps.
Une durée manquante reste manquante. Une réparation de durée nulle n'est pas retirée.
Le modèle CSV téléchargeable contient seulement les en-têtes : aucune donnée d'exemple n'est injectée dans l'étude.

Classeur du projet : feuille `events_history`, avec éventuellement `asset_info`, `analysis_settings`, `thermal_timeseries`, `thermal_params` et `maintenance_policies`. Les autres feuilles ne sont pas importées dans ce parcours.

CSV d'intervalles : `equipment_code`, `ttf_h`, et éventuellement `duree_rep_h`. Choisir explicitement la base de temps. Sans événements datés, le logiciel ne propose pas de date d'intervention.

Après importation, vérifier le rapport de contrôle. Les pannes et les intervalles sont comptés séparément. Les fenêtres peuvent être ajustées par équipement. Une fin d'observation postérieure à la dernière panne ne doit être renseignée que si la surveillance de cette période est documentée.

## Parcours de travail

### Indicateurs

Choisir l'équipement et la règle de sélection. AICc, prévision séquentielle et scénario explicite répondent à des questions distinctes. Le choix d'une ligne dans le sélecteur de scénario n'a d'effet que si la règle « Scénario explicite » est sélectionnée.

Le calcul compare HPP, PLP-NHPP, renouvellements Weibull et lognormal, et GRP Kijima I. Le bootstrap est désactivé par défaut pour éviter un calcul long involontaire. Les options proposent 300, 1 000 ou 3 000 réplications sur le modèle sélectionné. Pour un résultat de recherche, conserver les réglages, la graine et les limites de validation dans l'export.

Les colonnes KS/CvM nominales ne constituent pas une validation après estimation. Le statut bootstrap apparaît séparément. Une absence de rejet n'est pas une preuve de mécanisme physique ; un modèle rejeté reste signalé comme exploratoire.

### Optimisation

Les valeurs sont calculées depuis le temps de référence de l'analyse. Les coûts préventif et correctif doivent employer la même unité ou la même normalisation. La cible principale et la cible économique restent distinctes, et l'horizon proposé respecte les deux.

Le critère économique principal est implémenté pour PLP/HPP : `[Cp + Cc*(M(s+u)-M(s))]/u`. Pour les modèles de renouvellement et le GRP, l'horizon fiabiliste est calculé mais le coût économique reste indisponible. Choisir un PLP uniquement pour rendre un calcul économique possible n'en prouve pas l'adéquation ; un scénario explicite doit être justifié.

### Maintenance

L'échéance découle de la référence historique et de l'horizon. Elle ne glisse pas automatiquement avec la date du jour. Les jours restants sont une différence entre dates calendaires.

L'utilisateur choisit une action, une priorité et une justification. L'enregistrement produit une proposition à valider, pas un ordre de travail ni une notification. La page ne synchronise pas cette proposition avec les anciens modules de kits, stocks ou ordres de maintenance.

### Résultat global

Les exports Excel, JSON, CSV et PDF reprennent les résultats de la session sans réajuster les modèles. Le PDF est une synthèse textuelle ; les tableaux détaillés et les courbes numériques sont dans Excel/JSON. Les analyses, optimisations et propositions sont conservées dans la session Streamlit : télécharger les exports avant de fermer ou redémarrer celle-ci.

Un changement de données ou de fenêtre invalide les analyses, optimisations et propositions de la session. Un nouveau calcul d'analyse invalide les propositions et optimisations du même équipement.

## Vérifications réalisées

Tests effectués avec le système de test Streamlit `AppTest`, dans un projet temporaire utilisant les composants de connexion et de navigation existants. Le contexte de connexion est injecté uniquement dans le test ; les pages livrées appellent toujours `require_login()`.

- Ouverture des cinq pages sans exception.
- Sélection prédictive et calcul du cas de Funa : PLP retenu par le score séquentiel.
- Horizon PLP à 80 % : environ 1 813,4 h ; horizon économique à 70 % : environ 2 894,6 h, avec Cp=1 et Cc=5.
- Enregistrement d'une proposition de maintenance motivée.
- Génération et relecture des exports Excel et PDF ; sérialisation JSON stricte.
- Mise à jour de la fin d'observation et invalidation des résultats précédents.
- Les tests du moteur réalisés auparavant vérifient également les limites Kijima q=0/q=1, la censure, les données invalides et les horizons des cinq modèles.

Environnement de test : Python 3.12, Streamlit 1.63.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, openpyxl 3.1.5, ReportLab 4.4.9.

Les tests AppTest vérifient l'exécution et les interactions, pas le rendu visuel dans votre navigateur ni le fonctionnement des pages hors périmètre. L'importation CSV/Excel passe par pandas et les contrôles du datahub ; le téléversement via navigateur devra être vérifié avec votre fichier réel.
