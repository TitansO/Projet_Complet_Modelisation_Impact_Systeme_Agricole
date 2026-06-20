# =============================================================================
# Climat IA — API FastAPI Asynchrone (Version Production Corrigée)
# =============================================================================
# Logique de cache-aside Redis → PostgreSQL pour l'endpoint /predict.
# Architecture : Cache Miss → interrogation PostgreSQL + écriture Redis (TTL 3600s)
#                Cache Hit  → réponse directe Redis (latence < 1ms)
# =============================================================================

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import asyncpg
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration et Environnement
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())
logger = logging.getLogger("climat_ia_api")

POSTGRES_DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER','climat_user')}"
    f":{os.getenv('POSTGRES_PASSWORD','Cl1mat_S3cur3_2025!')}"
    f"@{os.getenv('POSTGRES_HOST','base_donnees_postgres')}"
    f":{os.getenv('POSTGRES_PORT','5432')}"
    f"/{os.getenv('POSTGRES_DB','climat_ia_db')}"
)

REDIS_HOST     = os.getenv("REDIS_HOST", "cache_redis")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "RedisClim@t2025!")
REDIS_TTL      = int(os.getenv("REDIS_TTL_SECONDS", 3600))

SYSTEMES_VALIDES = frozenset({
    "Systeme_Grandes_Cultures",
    "Systeme_Polyculture_Elevage",
    "Systeme_Agroecologique",
})

# ─────────────────────────────────────────────────────────────────────────────
# Modèles Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    systeme_production:  str   = Field(..., description="Système agricole TFE")
    scenario_rcp:        str   = Field("RCP_4.5", description="RCP_4.5 ou RCP_8.5")
    temperature_c:       float = Field(..., ge=-20, le=60,   description="Température °C")
    pluviometrie_mm:     float = Field(..., ge=0,   le=5000, description="Pluviométrie mm/an")
    indice_secheresse:   float = Field(..., ge=0,   le=1,    description="Indice sécheresse [0-1]")
    radiation_kwh:       float = Field(..., ge=0,   le=400,  description="Radiation solaire kWh/m²")

    @field_validator("systeme_production")
    @classmethod
    def valider_systeme(cls, v: str) -> str:
        if v not in SYSTEMES_VALIDES:
            raise ValueError(f"Système inconnu. Valeurs acceptées : {SYSTEMES_VALIDES}")
        return v

    @field_validator("scenario_rcp")
    @classmethod
    def valider_scenario(cls, v: str) -> str:
        if v not in {"RCP_4.5", "RCP_8.5"}:
            raise ValueError("scenario_rcp doit être 'RCP_4.5' ou 'RCP_8.5'")
        return v


class PredictionResponse(BaseModel):
    systeme_production:  str
    scenario_rcp:        str
    cluster_id:          int
    profil_climatique:   str
    impact_rendement:    str
    temperature_c_moy:   float
    pluviometrie_mm_moy: float
    score_silhouette:    Optional[float] = None
    cache_hit:           bool
    latence_ms:          int


# ─────────────────────────────────────────────────────────────────────────────
# État global de l'application (Connexions persistantes)
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    pg_pool:      Optional[asyncpg.Pool]   = None
    redis_client: Optional[aioredis.Redis] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et fermeture sécurisée des connexions réseau."""
    # ── Connexion PostgreSQL avec retry automatique au clonage ────────────
    logger.info("Connexion au pool PostgreSQL...")
    for attempt in range(1, 6):
        try:
            app_state.pg_pool = await asyncpg.create_pool(
                dsn=POSTGRES_DSN,
                min_size=2,
                max_size=10,
                command_timeout=30,
                max_inactive_connection_lifetime=300,
            )
            logger.info("Pool PostgreSQL établi avec succès.")
            break
        except Exception as e:
            if attempt == 5:
                logger.error("Impossible de joindre PostgreSQL après 5 tentatives. Abandon.")
                raise e
            logger.warning(f"PostgreSQL indisponible (Essai {attempt}/5)... Nouvelle tentative dans 3s")
            await time.sleep(3)

    # Vérification/Création de secours de la table des profils si absente au clonage
    async with app_state.pg_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profils_agronomiques (
                systeme_production VARCHAR(50),
                scenario_rcp VARCHAR(10),
                cluster_id INT,
                profil_climatique TEXT,
                impact_rendement TEXT,
                temperature_c_moy NUMERIC,
                pluviometrie_mm_moy NUMERIC,
                score_silhouette NUMERIC,
                PRIMARY KEY (systeme_production, scenario_rcp, cluster_id)
            );
        """)
        
        # Sécurisation de la table de logs d'inférence
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS logs_inference (
                id SERIAL PRIMARY KEY,
                systeme_production VARCHAR(50),
                scenario_rcp VARCHAR(10),
                temperature_c NUMERIC,
                pluviometrie_mm NUMERIC,
                indice_secheresse NUMERIC,
                radiation_kwh NUMERIC,
                cluster_id_predit INT,
                impact_predit TEXT,
                cache_hit BOOLEAN,
                latence_ms INT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)

    # ── Connexion Redis ───────────────────────────────────────────────────
    logger.info("Connexion au cache Redis...")
    try:
        app_state.redis_client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        await app_state.redis_client.ping()
        logger.info("Cache Redis opérationnel.")
    except Exception as exc:
        logger.error(f"Démarrage dégradé : Redis injoignable ({exc}). L'API utilisera PostgreSQL uniquement.")

    yield  # ← L'API gère activement les requêtes du serveur HTTP

    # ── Nettoyage propre à l'arrêt ────────────────────────────────────────
    if app_state.pg_pool:
        await app_state.pg_pool.close()
    if app_state.redis_client:
        await app_state.redis_client.aclose()
    logger.info("Toutes les connexions de l'infrastructure ont été fermées.")


# ─────────────────────────────────────────────────────────────────────────────
# Application FastAPI Principal
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Climat IA — API de Vulnérabilité Agronomique",
    description=(
        "API asynchrone exposant les prédictions de clusters K-Means "
        "pour les 3 systèmes de production agricole (TFE Delandmeter, 2021). "
        "Implémente le pattern cache-aside Redis → PostgreSQL."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires — Cache Réseau
# ─────────────────────────────────────────────────────────────────────────────
def _build_cache_key(req: PredictionRequest) -> str:
    """Génère une clé de hachage Redis distribuée."""
    t_bucket  = round(req.temperature_c)
    p_bucket  = int(req.pluviometrie_mm // 50) * 50
    s_bucket  = round(req.indice_secheresse, 1)
    return f"predict:{req.systeme_production}:{req.scenario_rcp}:{t_bucket}:{p_bucket}:{s_bucket}"


async def _get_from_cache(key: str) -> Optional[dict]:
    if not app_state.redis_client:
        return None
    try:
        raw = await app_state.redis_client.get(key)
        if raw:
            return json.loads(raw)
        return None
    except Exception as exc:
        logger.warning("Erreur de lecture Redis (Cache Miss forcé) : %s", exc)
        return None


async def _set_in_cache(key: str, data: dict) -> None:
    if not app_state.redis_client:
        return
    try:
        await app_state.redis_client.setex(key, REDIS_TTL, json.dumps(data))
    except Exception as exc:
        logger.warning("Échec de l'écriture asynchrone dans Redis : %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Classification Euclidienne K-Means en mémoire
# ─────────────────────────────────────────────────────────────────────────────
_CENTROÏDES = np.array([
    [27.0, 1450.0, 0.18, 95.0],    # Cluster 0 — Tropical humide
    [36.0,  280.0, 0.82, 210.0],   # Cluster 1 — Aride chaud (Sahel)
    [16.0,  680.0, 0.35, 155.0],   # Cluster 2 — Tempéré méditerranéen
    [29.0,  420.0, 0.58, 175.0],   # Cluster 3 — Semi-aride transitoire
])

_MU    = np.array([27.0, 707.5, 0.4825, 158.75])
_SIGMA = np.array([7.5,  477.0, 0.265,  46.00])


def _predict_cluster(temperature: float, pluvio: float, secheresse: float, radiation: float) -> int:
    x = np.array([temperature, pluvio, secheresse, radiation])
    x_norm = (x - _MU) / _SIGMA
    centroïdes_norm = (_CENTROÏDES - _MU) / _SIGMA
    distances = np.linalg.norm(centroïdes_norm - x_norm, axis=1)
    return int(np.argmin(distances))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint : POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_vulnerabilite(req: PredictionRequest) -> PredictionResponse:
    t_start = time.monotonic()
    cache_key = _build_cache_key(req)

    # 1. Traitement du Cache (Redis Hit)
    cached = await _get_from_cache(cache_key)
    if cached:
        latence = int((time.monotonic() - t_start) * 1000)
        cached["cache_hit"] = True
        cached["latence_ms"] = latence
        return PredictionResponse(**cached)

    # 2. Inférence Locale K-Means
    cluster_id = _predict_cluster(
        req.temperature_c, req.pluviometrie_mm,
        req.indice_secheresse, req.radiation_kwh,
    )

    # 3. Interrogation de la ressource PostgreSQL
    query = """
        SELECT cluster_id, profil_climatique, impact_rendement,
               temperature_c_moy, pluviometrie_mm_moy, score_silhouette
        FROM profils_agronomiques
        WHERE systeme_production = $1
          AND scenario_rcp = $2
          AND cluster_id = $3
        LIMIT 1
    """
    row = None
    try:
        async with app_state.pg_pool.acquire() as conn:
            row = await conn.fetchrow(query, req.systeme_production, req.scenario_rcp, cluster_id)
    except Exception as db_err:
        logger.error(f"Erreur d'exécution SQL : {db_err}")
        raise HTTPException(status_code=500, detail="Erreur interne d'accès à la base de données.")

    # Fallback intelligent si le Worker n'a pas encore injecté les profils (Évite le crash post-clonage)
    if not row:
        logger.warning(f"Profils non encore générés en base pour le cluster {cluster_id}. Génération d'une réponse par défaut.")
        row = {
            "cluster_id": cluster_id,
            "profil_climatique": f"Cluster Temporaire {cluster_id} (Données en cours de calcul)",
            "impact_rendement": "En attente d'analyse",
            "temperature_c_moy": req.temperature_c,
            "pluviometrie_mm_moy": req.pluviometrie_mm,
            "score_silhouette": 0.0
        }

    # 4. Consolidation des données de sortie
    result_data = {
        "systeme_production":  req.systeme_production,
        "scenario_rcp":        req.scenario_rcp,
        "cluster_id":          int(row["cluster_id"]),
        "profil_climatique":   row["profil_climatique"],
        "impact_rendement":    row["impact_rendement"],
        "temperature_c_moy":   float(row["temperature_c_moy"]),
        "pluviometrie_mm_moy": float(row["pluviometrie_mm_moy"]),
        "score_silhouette":    float(row["score_silhouette"]) if row["score_silhouette"] else None,
        "cache_hit":           False,
    }

    # 5. Écriture non-bloquante dans le cache Redis
    await _set_in_cache(cache_key, result_data)

    # 6. Journalisation asynchrone des métriques d'inférence (Sûreté de l'état)
    latence = int((time.monotonic() - t_start) * 1000)
    log_query = """
        INSERT INTO logs_inference
            (systeme_production, scenario_rcp, temperature_c, pluviometrie_mm,
             indice_secheresse, radiation_kwh, cluster_id_predit,
             impact_predit, cache_hit, latence_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """
    try:
        async with app_state.pg_pool.acquire() as conn:
            await conn.execute(
                log_query,
                req.systeme_production, req.scenario_rcp,
                req.temperature_c, req.pluviometrie_mm,
                req.indice_secheresse, req.radiation_kwh,
                cluster_id, result_data["impact_rendement"], False, latence,
            )
    except Exception as exc:
        logger.warning("Échec d'enregistrement du log d'inférence : %s", exc)

    result_data["latence_ms"] = latence
    return PredictionResponse(**result_data)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints Annexes (Monitoring & Diagnostics)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    """Vérifie l'état de santé de la passerelle et de ses dépendances."""
    health = {"status": "ok", "api": "up", "postgres": "unknown", "redis": "unknown"}

    if app_state.pg_pool:
        try:
            async with app_state.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health["postgres"] = "up"
        except Exception as exc:
            health["postgres"] = f"down: {exc}"
            health["status"] = "degraded"
    else:
        health["postgres"] = "pool_uninitialized"
        health["status"] = "degraded"

    if app_state.redis_client:
        try:
            await app_state.redis_client.ping()
            health["redis"] = "up"
        except Exception as exc:
            health["redis"] = f"down: {exc}"
    else:
        health["redis"] = "disabled"

    return health


@app.get("/profils")
async def liste_profils(systeme: Optional[str] = None, scenario: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retourne les profils disponibles stockés de manière relationnelle."""
    query = """
        SELECT systeme_production, scenario_rcp, cluster_id,
               profil_climatique, impact_rendement,
               temperature_c_moy, pluviometrie_mm_moy, score_silhouette
        FROM profils_agronomiques
        WHERE ($1::TEXT IS NULL OR systeme_production = $1)
          AND ($2::TEXT IS NULL OR scenario_rcp = $2)
        ORDER BY systeme_production, scenario_rcp, cluster_id
    """
    if not app_state.pg_pool:
        raise HTTPException(status_code=503, detail="Base de données non initialisée.")
        
    async with app_state.pg_pool.acquire() as conn:
        rows = await conn.fetch(query, systeme, scenario)
    return [dict(r) for r in rows]
