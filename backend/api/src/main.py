# =============================================================================
# Climat IA — API FastAPI Asynchrone (Version Enterprise Intégrée)
# =============================================================================
# Logique de cache-aside Redis → PostgreSQL pour l'endpoint /predict.
# Architecture : Cache Miss → interrogation PostgreSQL + écriture Redis (TTL 3600s)
#                Cache Hit  → réponse directe Redis (latence < 1ms)
# Complètement aligné sur le modèle relationnel ACID et NoSQL séries temporelles.
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
# Modèles Pydantic (Entrées/Sorties Sécurisées)
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
# État global de l'application (connexions persistantes)
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    pg_pool:      Optional[asyncpg.Pool]   = None
    redis_client: Optional[aioredis.Redis] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation asynchrone des connexions et des structures de données."""
    # ── Connexion PostgreSQL avec résilience au démarrage (Retry Loop) ────
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
            logger.info("Pool PostgreSQL établi.")
            break
        except Exception as e:
            if attempt == 5:
                logger.error("Défaut réseau persistant vers PostgreSQL. Abandon.")
                raise e
            logger.warning(f"PostgreSQL injoignable (Essai {attempt}/5). Nouvelle tentative dans 3s...")
            await time.sleep(3)

    # ── Initialisation sécurisée du schéma relationnel (Si absent au clonage) ──
    async with app_state.pg_pool.acquire() as conn:
        logger.info("Validation du schéma de base de données...")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        
        # Injection sécurisée des types Énumérés s'ils n'existent pas
        await conn.execute("""
            DO $$ BEGIN
                CREATE TYPE systeme_production_type AS ENUM ('Systeme_Grandes_Cultures', 'Systeme_Polyculture_Elevage', 'Systeme_Agroecologique');
            EXCEPTION WHEN duplicate_object THEN NULL; END $$;
            DO $$ BEGIN
                CREATE TYPE scenario_rcp_type AS ENUM ('RCP_4.5', 'RCP_8.5');
            EXCEPTION WHEN duplicate_object THEN NULL; END $$;
            DO $$ BEGIN
                CREATE TYPE impact_rendement_type AS ENUM ('HAUSSE', 'STABLE', 'BAISSE', 'CRITIQUE');
            EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        """)
        
        # Création des tables maîtresses conformes à votre architecture
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profils_agronomiques (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                systeme_production systeme_production_type NOT NULL,
                scenario_rcp scenario_rcp_type NOT NULL,
                cluster_id SMALLINT NOT NULL CHECK (cluster_id BETWEEN 0 AND 9),
                profil_climatique VARCHAR(64) NOT NULL,
                impact_rendement impact_rendement_type NOT NULL,
                temperature_c_moy NUMERIC(5,2) NOT NULL,
                pluviometrie_mm_moy NUMERIC(7,2) NOT NULL,
                indice_secheresse_moy NUMERIC(4,3) NOT NULL,
                radiation_kwh_moy NUMERIC(6,2) NOT NULL,
                score_silhouette NUMERIC(4,3),
                periode_debut SMALLINT NOT NULL,
                periode_fin SMALLINT NOT NULL,
                nb_observations INTEGER NOT NULL DEFAULT 0,
                cree_le TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                mis_a_jour_le TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT uq_profil_systeme_scenario_cluster UNIQUE (systeme_production, scenario_rcp, cluster_id)
            );
            
            CREATE TABLE IF NOT EXISTS logs_inference (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                systeme_production systeme_production_type NOT NULL,
                scenario_rcp scenario_rcp_type,
                temperature_c NUMERIC(5,2),
                pluviometrie_mm NUMERIC(7,2),
                indice_secheresse NUMERIC(4,3),
                radiation_kwh NUMERIC(6,2),
                cluster_id_predit SMALLINT,
                impact_predit impact_rendement_type,
                cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                latence_ms INTEGER,
                demande_le TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        
        # Indexation
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profils_systeme ON profils_agronomiques(systeme_production);
            CREATE INDEX IF NOT EXISTS idx_logs_systeme ON logs_inference(systeme_production);
        """)

        # Population des données initiales par défaut si la table est vide au premier démarrage
        count = await conn.fetchval("SELECT COUNT(*) FROM profils_agronomiques;")
        if count == 0:
            logger.info("Ingestion des 4 profils agronomiques de référence (TFE Delandmeter)...")
            await conn.execute("""
                INSERT INTO profils_agronomiques (
                    systeme_production, scenario_rcp, cluster_id, profil_climatique, impact_rendement,
                    temperature_c_moy, pluviometrie_mm_moy, indice_secheresse_moy, radiation_kwh_moy,
                    score_silhouette, periode_debut, periode_fin, nb_observations
                ) VALUES 
                ('Systeme_Grandes_Cultures'::systeme_production_type, 'RCP_4.5'::scenario_rcp_type, 0, 'Tropical humide', 'HAUSSE'::impact_rendement_type, 27.0, 1450.0, 0.18, 95.0, 0.72, 2045, 2069, 120),
                ('Systeme_Grandes_Cultures'::systeme_production_type, 'RCP_4.5'::scenario_rcp_type, 1, 'Aride chaud (Sahel)', 'CRITIQUE'::impact_rendement_type, 36.0, 280.0, 0.82, 210.0, 0.68, 2045, 2069, 120),
                ('Systeme_Grandes_Cultures'::systeme_production_type, 'RCP_4.5'::scenario_rcp_type, 2, 'Tempéré méditerranéen', 'STABLE'::impact_rendement_type, 16.0, 680.0, 0.35, 155.0, 0.61, 2045, 2069, 120),
                ('Systeme_Grandes_Cultures'::systeme_production_type, 'RCP_4.5'::scenario_rcp_type, 3, 'Semi-aride transitoire', 'BAISSE'::impact_rendement_type, 29.0, 420.0, 0.58, 175.0, 0.55, 2045, 2069, 120)
                ON CONFLICT DO NOTHING;
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
        logger.warning(f"Mode dégradé : Cache hors-ligne ({exc}). Inférence directe via PostgreSQL active.")

    yield  # ← Fonctionnement nominal de l'API Gateway

    # ── Fermeture Propre des Pools ────────────────────────────────────────
    if app_state.pg_pool:
        await app_state.pg_pool.close()
    if app_state.redis_client:
        await app_state.redis_client.aclose()
    logger.info("Ressources réseau libérées proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# Application FastAPI Setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Climat IA — API de Vulnérabilité Agronomique",
    description="Passerelle d'inférence distribuée à faible latence (Pattern Cache-Aside)",
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
# Gestionnaires de Cache
# ─────────────────────────────────────────────────────────────────────────────
def _build_cache_key(req: PredictionRequest) -> str:
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
        logger.warning("Cache Miss forcé (Redis inaccessible) : %s", exc)
        return None


async def _set_in_cache(key: str, data: dict) -> None:
    if not app_state.redis_client:
        return
    try:
        await app_state.redis_client.setex(key, REDIS_TTL, json.dumps(data))
    except Exception as exc:
        logger.warning("Échec écriture cache Redis : %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Moteur d'Inférence Numérique (K-Means en mémoire)
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
# Endpoint Principal : POST /predict
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_vulnerabilite(req: PredictionRequest) -> PredictionResponse:
    t_start = time.monotonic()
    cache_key = _build_cache_key(req)

    # Étape 1 : Interrogation du cache Redis (Cache Hit)
    cached = await _get_from_cache(cache_key)
    if cached:
        latence = int((time.monotonic() - t_start) * 1000)
        cached["cache_hit"] = True
        cached["latence_ms"] = latence
        return PredictionResponse(**cached)

    # Étape 2 : Classification Euclidienne locale via les centroïdes
    cluster_id = _predict_cluster(
        req.temperature_c, req.pluviometrie_mm,
        req.indice_secheresse, req.radiation_kwh,
    )

    # Étape 3 : Requête PostgreSQL (Cast explicite des types énumérés en TEXT pour compatibilité python)
    query = """
        SELECT cluster_id, profil_climatique, impact_rendement::TEXT as impact_rendement,
               temperature_c_moy, pluviometrie_mm_moy, score_silhouette
        FROM profils_agronomiques
        WHERE systeme_production = $1::systeme_production_type
          AND scenario_rcp = $2::scenario_rcp_type
          AND cluster_id = $3
        LIMIT 1
    """
    
    row = None
    async with app_state.pg_pool.acquire() as conn:
        row = await conn.fetchrow(query, req.systeme_production, req.scenario_rcp, cluster_id)

    # Fallback applicatif si aucun profil n'est retourné (Ex: cluster 1 sous RCP 8.5 non encore calculé)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_444_RESPONSE_VALUE_MISSING if hasattr(status, 'HTTP_444_RESPONSE_VALUE_MISSING') else 404,
            detail=f"Profil agronomique non calculé pour {req.systeme_production} / {req.scenario_rcp} / Cluster {cluster_id}."
        )

    # Étape 4 : Consolidation de la réponse et conversion stricte des types numériques Decimal (NUMERIC) -> float
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

    # Étape 5 : Persistance asynchrone dans Redis
    await _set_in_cache(cache_key, result_data)

    # Étape 6 : Écriture asynchrone des métriques d'audit d'inférence (Fire-and-forget)
    latence = int((time.monotonic() - t_start) * 1000)
    log_query = """
        INSERT INTO logs_inference
            (systeme_production, scenario_rcp, temperature_c, pluviometrie_mm,
             indice_secheresse, radiation_kwh, cluster_id_predit,
             impact_predit, cache_hit, latence_ms)
        VALUES ($1::systeme_production_type, $2::scenario_rcp_type, $3, $4, $5, $6, $7, $8::impact_rendement_type, $9, $10)
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
        logger.warning(f"Journalisation d'inférence en échec : {exc}")

    result_data["latence_ms"] = latence
    return PredictionResponse(**result_data)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint : Diagnostics Métiers
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
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
    except Exception:
        health["redis"] = "down"
    return health
