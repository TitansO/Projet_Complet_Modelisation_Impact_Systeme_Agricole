"""
Climat IA — API FastAPI Asynchrone
====================================
Logique de cache-aside Redis → PostgreSQL pour l'endpoint /predict.
Architecture : Cache Miss → interrogation PostgreSQL + écriture Redis (TTL 3600s)
               Cache Hit  → réponse directe Redis (latence < 1ms)

Systèmes agricoles (TFE Delandmeter, 2021) :
  - Systeme_Grandes_Cultures   (BAU)
  - Systeme_Polyculture_Elevage (Vegan)
  - Systeme_Agroecologique      (ICLS — haute résilience)
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import asyncpg
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
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
    score_silhouette:    Optional[float]
    cache_hit:           bool
    latence_ms:          int


# ─────────────────────────────────────────────────────────────────────────────
# État global de l'application (connexions persistantes)
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    pg_pool:    Optional[asyncpg.Pool]         = None
    redis_client: Optional[aioredis.Redis]     = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation et fermeture des connexions au démarrage/arrêt de l'API."""
    # ── Connexion PostgreSQL ──────────────────────────────────────────────
    logger.info("Connexion au pool PostgreSQL…")
    app_state.pg_pool = await asyncpg.create_pool(
        dsn=POSTGRES_DSN,
        min_size=2,
        max_size=10,
        command_timeout=30,
        max_inactive_connection_lifetime=300,
    )
    logger.info("Pool PostgreSQL établi.")

    # ── Connexion Redis ───────────────────────────────────────────────────
    logger.info("Connexion au cache Redis…")
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

    yield  # ← L'API est en service

    # ── Nettoyage à l'arrêt ───────────────────────────────────────────────
    await app_state.pg_pool.close()
    await app_state.redis_client.aclose()
    logger.info("Connexions fermées proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# Application FastAPI
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
# Utilitaires — Cache Redis
# ─────────────────────────────────────────────────────────────────────────────
def _build_cache_key(req: PredictionRequest) -> str:
    """
    Clé Redis déterministe pour une requête de prédiction.
    Granularité : (système, scénario, température arrondie à 1°C,
                   pluviométrie arrondie à 50mm, indice sécheresse arrondi à 0.1)
    Cette granularité permet la réutilisation du cache pour des requêtes
    climatiquement équivalentes sans multiplication excessive des clés.
    """
    t_bucket  = round(req.temperature_c)
    p_bucket  = int(req.pluviometrie_mm // 50) * 50
    s_bucket  = round(req.indice_secheresse, 1)
    return (
        f"predict:"
        f"{req.systeme_production}:"
        f"{req.scenario_rcp}:"
        f"{t_bucket}:"
        f"{p_bucket}:"
        f"{s_bucket}"
    )


async def _get_from_cache(key: str) -> Optional[dict]:
    """Lecture depuis Redis. Retourne None en cas de Cache Miss ou d'erreur."""
    try:
        raw = await app_state.redis_client.get(key)
        if raw:
            logger.debug("Cache HIT pour clé : %s", key)
            return json.loads(raw)
        logger.debug("Cache MISS pour clé : %s", key)
        return None
    except Exception as exc:
        # Le cache Redis est optionnel — une panne ne doit pas briser l'API
        logger.warning("Erreur Redis (cache miss forcé) : %s", exc)
        return None


async def _set_in_cache(key: str, data: dict) -> None:
    """Écriture dans Redis avec TTL. Silencieuse en cas d'erreur."""
    try:
        await app_state.redis_client.setex(key, REDIS_TTL, json.dumps(data))
        logger.debug("Cache SET clé %s (TTL %ds)", key, REDIS_TTL)
    except Exception as exc:
        logger.warning("Échec écriture Redis (non bloquant) : %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires — K-Means en mémoire (classification par distance euclidienne)
# ─────────────────────────────────────────────────────────────────────────────
# Centroïdes pré-entraînés (normalisés Z-score, μ et σ calculés sur le dataset)
_CENTROÏDES = np.array([
    [27.0, 1450.0, 0.18, 95.0],    # Cluster 0 — Tropical humide
    [36.0,  280.0, 0.82, 210.0],   # Cluster 1 — Aride chaud (Sahel)
    [16.0,  680.0, 0.35, 155.0],   # Cluster 2 — Tempéré méditerranéen
    [29.0,  420.0, 0.58, 175.0],   # Cluster 3 — Semi-aride transitoire
])

_MU    = np.array([27.0, 707.5, 0.4825, 158.75])
_SIGMA = np.array([7.5,  477.0, 0.265,  46.00])


def _predict_cluster(temperature: float, pluvio: float,
                     secheresse: float, radiation: float) -> int:
    """Classification par distance euclidienne aux centroïdes (O(k×d))."""
    x = np.array([temperature, pluvio, secheresse, radiation])
    x_norm = (x - _MU) / _SIGMA
    centroïdes_norm = (_CENTROÏDES - _MU) / _SIGMA
    distances = np.linalg.norm(centroïdes_norm - x_norm, axis=1)
    return int(np.argmin(distances))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint : POST /predict
# Pattern cache-aside : Redis → PostgreSQL → Redis (mise en cache du résultat)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_vulnerabilite(req: PredictionRequest) -> PredictionResponse:
    """
    Prédiction du profil de vulnérabilité agronomique.

    Logique cache-aside :
    1. Calcul de la clé Redis déterministe pour la requête.
    2. Vérification du cache Redis (GET).
       → Cache HIT : retour immédiat depuis Redis (latence < 1ms).
       → Cache MISS : classification K-Means + requête PostgreSQL + écriture Redis.
    3. Enregistrement du log d'inférence dans PostgreSQL (non bloquant).
    """
    t_start = time.monotonic()
    cache_key = _build_cache_key(req)

    # ─── Étape 1 : Vérification Redis ────────────────────────────────────
    cached = await _get_from_cache(cache_key)
    if cached:
        latence = int((time.monotonic() - t_start) * 1000)
        cached["cache_hit"] = True
        cached["latence_ms"] = latence
        return PredictionResponse(**cached)

    # ─── Étape 2 : Classification K-Means ────────────────────────────────
    cluster_id = _predict_cluster(
        req.temperature_c, req.pluviometrie_mm,
        req.indice_secheresse, req.radiation_kwh,
    )

    # ─── Étape 3 : Interrogation PostgreSQL ──────────────────────────────
    query = """
        SELECT cluster_id, profil_climatique, impact_rendement,
               temperature_c_moy, pluviometrie_mm_moy, score_silhouette
        FROM profils_agronomiques
        WHERE systeme_production = $1
          AND scenario_rcp = $2
          AND cluster_id = $3
        LIMIT 1
    """
    async with app_state.pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            query, req.systeme_production, req.scenario_rcp, cluster_id
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Aucun profil trouvé pour {req.systeme_production} / "
                f"{req.scenario_rcp} / cluster {cluster_id}. "
                "Exécutez d'abord le worker de clustering."
            ),
        )

    # ─── Étape 4 : Construction de la réponse ────────────────────────────
    result_data = {
        "systeme_production":  req.systeme_production,
        "scenario_rcp":        req.scenario_rcp,
        "cluster_id":          row["cluster_id"],
        "profil_climatique":   row["profil_climatique"],
        "impact_rendement":    row["impact_rendement"],
        "temperature_c_moy":   float(row["temperature_c_moy"]),
        "pluviometrie_mm_moy": float(row["pluviometrie_mm_moy"]),
        "score_silhouette":    float(row["score_silhouette"]) if row["score_silhouette"] else None,
        "cache_hit":           False,
    }

    # ─── Étape 5 : Mise en cache Redis ───────────────────────────────────
    await _set_in_cache(cache_key, result_data)

    # ─── Étape 6 : Log d'inférence PostgreSQL (fire-and-forget) ──────────
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
                cluster_id, row["impact_rendement"], False, latence,
            )
    except Exception as exc:
        logger.warning("Échec log inférence (non bloquant) : %s", exc)

    result_data["latence_ms"] = latence
    return PredictionResponse(**result_data)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint : GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    """Vérifie la connectivité PostgreSQL et Redis."""
    health = {"status": "ok", "api": "up", "postgres": "unknown", "redis": "unknown"}

    try:
        async with app_state.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health["postgres"] = "up"
    except Exception as exc:
        health["postgres"] = f"down: {exc}"
        health["status"] = "degraded"

    try:
        await app_state.redis_client.ping()
        health["redis"] = "up"
    except Exception as exc:
        health["redis"] = f"down: {exc}"
        # Redis non critique — l'API continue à fonctionner (cache miss total)

    return health


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint : GET /profils
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/profils")
async def liste_profils(
    systeme: Optional[str] = None,
    scenario: Optional[str] = None,
) -> list[dict]:
    """Liste les profils agronomiques disponibles dans PostgreSQL."""
    query = """
        SELECT systeme_production, scenario_rcp, cluster_id,
               profil_climatique, impact_rendement,
               temperature_c_moy, pluviometrie_mm_moy, score_silhouette
        FROM profils_agronomiques
        WHERE ($1::TEXT IS NULL OR systeme_production = $1)
          AND ($2::TEXT IS NULL OR scenario_rcp = $2)
        ORDER BY systeme_production, scenario_rcp, cluster_id
    """
    async with app_state.pg_pool.acquire() as conn:
        rows = await conn.fetch(query, systeme, scenario)
    return [dict(r) for r in rows]
