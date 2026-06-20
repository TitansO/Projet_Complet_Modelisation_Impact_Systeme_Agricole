# =============================================================================
# Dockerfile — API FastAPI Climat IA
# =============================================================================

# Étape 1 : Utilisation d'une image de base Python légère et stable
FROM python:3.11-slim

# Étape 2 : Définition des variables d'environnement de production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Étape 3 : Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Étape 4 : Installation des dépendances système requises pour la compilation (ex: numpy, asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Étape 5 : Copie et installation des dépendances Python
# (Assurez-vous d'avoir un fichier requirements.txt dans le même dossier)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Étape 6 : Copie de l'ensemble du code source de l'API
COPY src/ /app/src/

# Étape 7 : Sécurisation — Exécution du conteneur avec un utilisateur non-root
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Étape 8 : Exposition du port interne de FastAPI
EXPOSE 8000

# Étape 9 : Commande de lancement du serveur ASGI Uvicorn en mode asynchrone
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
