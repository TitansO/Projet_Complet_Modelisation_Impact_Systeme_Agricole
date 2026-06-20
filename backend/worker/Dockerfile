# =============================================================================
# Dockerfile — Workspace Data Science / Jupyter Climat IA
# =============================================================================

# Étape 1 : Utilisation de l'image officielle Jupyter Data Science (embarque déjà numpy et pandas)
FROM jupyter/datascience-notebook:python-3.11

# Étape 2 : Définition du répertoire de travail par défaut de Jupyter
WORKDIR /home/jovyan/work

# Étape 3 : Installation des librairies spécifiques requises pour Climat IA
# - scikit-learn : pour l'algorithme K-Means non supervisé
# - asyncpg : pour communiquer de manière asynchrone avec PostgreSQL
# - cassandra-driver : pour interagir avec le flux NoSQL des séries temporelles
RUN pip install --no-cache-dir \
    scikit-learn \
    asyncpg \
    cassandra-driver \
    redis

# Étape 4 : Copie des scripts d'initialisation, des notebooks ou des scripts de clustering
# (Si vous souhaitez embarquer des scripts pré-existants directement au build)
# COPY --chown=jovyan:users scripts/ /home/jovyan/work/scripts/

# Étape 5 : Exposition du port d'accès à l'interface web de Jupyter Notebook
EXPOSE 8888

# Le point d'entrée par défaut hérité de l'image jupyter lance automatiquement le serveur
