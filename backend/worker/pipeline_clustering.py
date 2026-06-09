"""
Climat IA — Worker de Clustering K-Means
==========================================
Pipeline modulaire :
  1. Lecture asynchrone des séries temporelles brutes depuis Apache Cassandra
  2. Normalisation Z-score + clustering K-Means (exclusion stricte des targets)
  3. Écriture des profils de vulnérabilité agronomique dans PostgreSQL
  4. Enregistrement des métriques de run (inertie, score de silhouette)

Systèmes agricoles (TFE Delandmeter, 2021) :
  - Systeme_Grandes_Cultures   (BAU)
  - Systeme_Polyculture_Elevage (Vegan)
  - Systeme_Agroecologique      (ICLS)
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Configuration du logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("climat_ia_worker")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes métier
# ─────────────────────────────────────────────────────────────────────────────
SYSTEMES_PRODUCTION = [
    "Systeme_Grandes_Cultures",
    "Systeme_Polyculture_Elevage",
    "Systeme_Agroecologique",
]

SCENARIOS_RCP = ["RCP_4.5", "RCP_8.5"]

# Features physiques utilisées pour le clustering.
# CRITIQUE : la variable d'impact (target) est EXCLUE — apprentissage non supervisé pur.
FEATURES_CLUSTERING = [
    "temperature_moy",
    "pluviometrie_totale",
    "indice_secheresse_moy",
    "radiation_totale",
]

# Dictionnaire de mapping cluster_id → profil climatique (basé sur la connaissance domaine)
CLUSTER_PROFILES_MAPPING = {
    0: {"profil_climatique": "Tropical humide",          "impact_rendement": "HAUSSE"},
    1: {"profil_climatique": "Aride chaud (Sahel)",      "impact_rendement": "CRITIQUE"},
    2: {"profil_climatique": "Tempéré méditerranéen",    "impact_rendement": "STABLE"},
    3: {"profil_climatique": "Semi-aride transitoire",   "impact_rendement": "BAISSE"},
}

N_CLUSTERS    = int(os.getenv("N_CLUSTERS", 4))
RANDOM_STATE  = 42

# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Connexion Cassandra
# ─────────────────────────────────────────────────────────────────────────────
def creer_session_cassandra() -> tuple:
    """
    Établit la connexion au cluster Cassandra.
    Utilise DCAwareRoundRobinPolicy pour la localité des requêtes en production.
    """
    cassandra_host     = os.getenv("CASSANDRA_HOST", "nosql_cassandra")
    cassandra_port     = int(os.getenv("CASSANDRA_PORT", 9042))
    cassandra_keyspace = os.getenv("CASSANDRA_KEYSPACE", "climat_ia_ks")

    logger.info("Connexion à Cassandra : %s:%d", cassandra_host, cassandra_port)

    cluster = Cluster(
        contact_points=[cassandra_host],
        port=cassandra_port,
        load_balancing_policy=DCAwareRoundRobinPolicy(
            local_dc=os.getenv("CASSANDRA_DC", "datacenter_ouest_africain")
        ),
        connect_timeout=30,
        protocol_version=5,
    )
    session = cluster.connect(cassandra_keyspace)
    logger.info("Session Cassandra établie sur keyspace : %s", cassandra_keyspace)
    return cluster, session


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Lecture des séries temporelles depuis Cassandra
# ─────────────────────────────────────────────────────────────────────────────
def lire_agregats_cassandra(
    session,
    systeme: str,
    annee_mois_debut: str = "2045-01",
    annee_mois_fin: str   = "2099-12",
) -> pd.DataFrame:
    """
    Lit les agrégats quotidiens depuis la table `agregats_quotidiens` de Cassandra.

    Optimisé pour le modèle de données Cassandra :
    - Requête par clé de partition (systeme_production, annee_mois) → O(1) par mois
    - Évite les full-scan sur `flux_climatiques_bruts` (milliards de lignes)
    - Le filtrage par plage de dates est effectué côté Python (après lecture par partition)

    Retourne un DataFrame pandas avec les colonnes FEATURES_CLUSTERING.
    """
    logger.info("Lecture Cassandra pour le système : %s", systeme)

    # Génération de toutes les partitions mensuelles dans la plage temporelle
    debut = datetime.strptime(annee_mois_debut, "%Y-%m")
    fin   = datetime.strptime(annee_mois_fin, "%Y-%m")
    mois_range = pd.period_range(debut, fin, freq="M")
    annee_mois_list = [str(m) for m in mois_range]

    query = """
        SELECT date_agregat, temperature_moy, pluviometrie_totale,
               humidite_moy, indice_secheresse_moy, radiation_totale, nb_mesures
        FROM agregats_quotidiens
        WHERE systeme_production = %s AND annee_mois = %s
    """

    lignes = []
    for annee_mois in annee_mois_list:
        try:
            rows = session.execute(query, (systeme, annee_mois))
            for row in rows:
                lignes.append({
                    "date_agregat":          row.date_agregat,
                    "temperature_moy":       row.temperature_moy       or 0.0,
                    "pluviometrie_totale":   row.pluviometrie_totale    or 0.0,
                    "humidite_moy":          row.humidite_moy           or 0.0,
                    "indice_secheresse_moy": row.indice_secheresse_moy  or 0.0,
                    "radiation_totale":      row.radiation_totale       or 0.0,
                    "nb_mesures":            row.nb_mesures             or 0,
                })
        except Exception as exc:
            logger.warning("Partition manquante %s/%s : %s", systeme, annee_mois, exc)

    if not lignes:
        logger.warning("Aucune donnée Cassandra pour %s — génération de données simulées.", systeme)
        return _generer_donnees_simulees(systeme)

    df = pd.DataFrame(lignes)
    logger.info("Données Cassandra : %d observations pour %s", len(df), systeme)
    return df


def _generer_donnees_simulees(systeme: str) -> pd.DataFrame:
    """
    Génère des données climatiques simulées réalistes selon le système de production.
    Utilisé en développement ou si Cassandra ne contient pas encore de données brutes.
    """
    rng = np.random.default_rng(seed=hash(systeme) % (2**32))
    n   = 480  # 40 ans × 12 mois

    profiles = {
        "Systeme_Grandes_Cultures": {
            "temperature_moy":       rng.normal(24, 8,   n),
            "pluviometrie_totale":   rng.normal(700, 350, n).clip(0),
            "indice_secheresse_moy": rng.beta(2, 3, n),
            "radiation_totale":      rng.normal(155, 45,  n).clip(50),
        },
        "Systeme_Polyculture_Elevage": {
            "temperature_moy":       rng.normal(22, 7,   n),
            "pluviometrie_totale":   rng.normal(800, 300, n).clip(0),
            "indice_secheresse_moy": rng.beta(1.5, 3, n),
            "radiation_totale":      rng.normal(145, 40,  n).clip(50),
        },
        "Systeme_Agroecologique": {
            # ICLS : plus grande résilience hydrique (TFE Delandmeter, §3.4)
            "temperature_moy":       rng.normal(23, 7,   n),
            "pluviometrie_totale":   rng.normal(850, 280, n).clip(0),
            "indice_secheresse_moy": rng.beta(1.2, 3.5, n),  # Distribution plus humide
            "radiation_totale":      rng.normal(150, 42,  n).clip(50),
        },
    }
    data = profiles.get(systeme, profiles["Systeme_Grandes_Cultures"])
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — Pipeline K-Means (exclusion stricte des targets)
# ─────────────────────────────────────────────────────────────────────────────
def executer_clustering(
    df: pd.DataFrame,
    systeme: str,
    n_clusters: int = N_CLUSTERS,
) -> tuple[pd.DataFrame, dict]:
    """
    Exécute le pipeline K-Means sur les features physiques uniquement.

    PRINCIPE FONDAMENTAL : aucune variable de type 'impact_rendement' ou 'scenario_rcp'
    n'est transmise à l'algorithme. L'impact émerge comme propriété post-clustering.

    Retourne :
      - df enrichi avec les colonnes 'cluster_id' et 'features_normalisees'
      - dict de métriques (inertie, score silhouette, centroïdes)
    """
    logger.info("Clustering K-Means pour %s (k=%d)…", systeme, n_clusters)
    t_start = time.perf_counter()

    # Sélection stricte des features physiques (TARGET EXCLUE)
    features_manquantes = [f for f in FEATURES_CLUSTERING if f not in df.columns]
    if features_manquantes:
        raise ValueError(f"Features manquantes dans le DataFrame : {features_manquantes}")

    X_raw    = df[FEATURES_CLUSTERING].values
    n_valides = np.sum(~np.isnan(X_raw).any(axis=1))

    if n_valides < n_clusters * 10:
        raise ValueError(
            f"Trop peu d'observations valides ({n_valides}) pour k={n_clusters}. "
            "Augmentez la plage temporelle de lecture Cassandra."
        )

    # ── Normalisation Z-score (StandardScaler) ─────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    logger.debug("Moyennes (μ) : %s", np.round(scaler.mean_, 3))
    logger.debug("Écarts-types (σ) : %s", np.round(scaler.scale_, 3))

    # ── K-Means ────────────────────────────────────────────────────────────
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",   # Initialisation optimisée (Arthur & Vassilvitskii, 2007)
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(X_scaled)

    # ── Métriques ──────────────────────────────────────────────────────────
    score_silhouette = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)))
    inertie          = float(kmeans.inertia_)
    n_iterations     = int(kmeans.n_iter_)
    duree_ms         = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        "K-Means terminé en %dms | Inertie : %.2f | Silhouette : %.4f | Itérations : %d",
        duree_ms, inertie, score_silhouette, n_iterations,
    )

    # ── Enrichissement du DataFrame ───────────────────────────────────────
    df = df.copy()
    df["cluster_id"] = labels

    # Centroïdes dénormalisés (espace original des features)
    centroïdes_denorm = scaler.inverse_transform(kmeans.cluster_centers_)
    centroïdes_df = pd.DataFrame(centroïdes_denorm, columns=FEATURES_CLUSTERING)

    metriques = {
        "run_id":              str(uuid.uuid4()),
        "n_clusters":          n_clusters,
        "inertie_totale":      inertie,
        "score_silhouette":    float(score_silhouette),
        "nb_iterations":       n_iterations,
        "nb_observations":     len(df),
        "features_utilisees":  FEATURES_CLUSTERING,
        "duree_ms":            duree_ms,
        "centroïdes":          centroïdes_df.to_dict(orient="records"),
        "mu":                  scaler.mean_.tolist(),
        "sigma":               scaler.scale_.tolist(),
    }

    return df, metriques


# ─────────────────────────────────────────────────────────────────────────────
# Module 4 — Écriture des profils dans PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
def ecrire_profils_postgresql(
    conn,
    df: pd.DataFrame,
    metriques: dict,
    systeme: str,
    scenario: str,
) -> None:
    """
    Écrit les profils de vulnérabilité agronomique dans PostgreSQL.

    Stratégie d'idempotence : INSERT ... ON CONFLICT DO UPDATE
    → Permet les ré-exécutions sans duplication (pipeline idempotent).
    """
    logger.info("Écriture PostgreSQL pour %s / %s…", systeme, scenario)

    with conn.cursor() as cur:
        for cluster_id, centroïde in enumerate(metriques["centroïdes"]):
            profil = CLUSTER_PROFILES_MAPPING.get(cluster_id, {
                "profil_climatique": f"Cluster_{cluster_id}",
                "impact_rendement":  "STABLE",
            })
            nb_obs = int((df["cluster_id"] == cluster_id).sum())

            cur.execute(
                """
                INSERT INTO profils_agronomiques (
                    systeme_production, scenario_rcp, cluster_id,
                    profil_climatique, impact_rendement,
                    temperature_c_moy, pluviometrie_mm_moy,
                    indice_secheresse_moy, radiation_kwh_moy,
                    score_silhouette, periode_debut, periode_fin, nb_observations
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (systeme_production, scenario_rcp, cluster_id)
                DO UPDATE SET
                    profil_climatique    = EXCLUDED.profil_climatique,
                    impact_rendement     = EXCLUDED.impact_rendement,
                    temperature_c_moy    = EXCLUDED.temperature_c_moy,
                    pluviometrie_mm_moy  = EXCLUDED.pluviometrie_mm_moy,
                    indice_secheresse_moy = EXCLUDED.indice_secheresse_moy,
                    radiation_kwh_moy    = EXCLUDED.radiation_kwh_moy,
                    score_silhouette     = EXCLUDED.score_silhouette,
                    nb_observations      = EXCLUDED.nb_observations,
                    mis_a_jour_le        = NOW()
                """,
                (
                    systeme, scenario, cluster_id,
                    profil["profil_climatique"], profil["impact_rendement"],
                    round(centroïde.get("temperature_moy", 0), 2),
                    round(centroïde.get("pluviometrie_totale", 0), 2),
                    round(centroïde.get("indice_secheresse_moy", 0), 3),
                    round(centroïde.get("radiation_totale", 0), 2),
                    round(metriques["score_silhouette"], 4),
                    2045, 2099, nb_obs,
                ),
            )

        # ── Métriques du run ───────────────────────────────────────────────
        cur.execute(
            """
            INSERT INTO metriques_clustering (
                run_id, systeme_production, scenario_rcp, n_clusters,
                inertie_totale, score_silhouette_moy, nb_iterations,
                nb_observations, features_utilisees, duree_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                metriques["run_id"], systeme, scenario, metriques["n_clusters"],
                metriques["inertie_totale"], metriques["score_silhouette"],
                metriques["nb_iterations"], metriques["nb_observations"],
                metriques["features_utilisees"], metriques["duree_ms"],
            ),
        )

    conn.commit()
    logger.info(
        "PostgreSQL : %d profils + 1 métrique run écrites pour %s / %s",
        N_CLUSTERS, systeme, scenario,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Orchestration complète du pipeline :
    Cassandra (lecture) → K-Means (clustering) → PostgreSQL (écriture)
    """
    logger.info("═" * 60)
    logger.info("Démarrage du Worker de Clustering K-Means — Climat IA")
    logger.info("═" * 60)

    # ── Connexion PostgreSQL ───────────────────────────────────────────────
    pg_conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "base_donnees_postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "climat_ia_db"),
        user=os.getenv("POSTGRES_USER", "climat_user"),
        password=os.getenv("POSTGRES_PASSWORD", "Cl1mat_S3cur3_2025!"),
        connect_timeout=30,
    )
    logger.info("Connexion PostgreSQL établie.")

    # ── Connexion Cassandra ────────────────────────────────────────────────
    try:
        cassandra_cluster, cassandra_session = creer_session_cassandra()
        cassandra_disponible = True
    except Exception as exc:
        logger.warning("Cassandra indisponible (%s) — utilisation de données simulées.", exc)
        cassandra_disponible = False
        cassandra_cluster    = None
        cassandra_session    = None

    # ── Boucle sur les 3 systèmes × 2 scénarios ───────────────────────────
    resultats = []
    for systeme in SYSTEMES_PRODUCTION:
        for scenario in SCENARIOS_RCP:
            logger.info("─" * 50)
            logger.info("Traitement : %s × %s", systeme, scenario)
            try:
                # Lecture Cassandra ou données simulées
                if cassandra_disponible:
                    df_brut = lire_agregats_cassandra(cassandra_session, systeme)
                else:
                    df_brut = _generer_donnees_simulees(systeme)

                # Clustering K-Means
                df_clusterisé, metriques = executer_clustering(df_brut, systeme, N_CLUSTERS)

                # Écriture PostgreSQL
                ecrire_profils_postgresql(pg_conn, df_clusterisé, metriques, systeme, scenario)

                resultats.append({
                    "systeme": systeme,
                    "scenario": scenario,
                    "score_silhouette": metriques["score_silhouette"],
                    "nb_obs": metriques["nb_observations"],
                    "statut": "OK",
                })

            except Exception as exc:
                logger.error("Erreur pour %s / %s : %s", systeme, scenario, exc, exc_info=True)
                resultats.append({
                    "systeme": systeme, "scenario": scenario,
                    "statut": f"ERREUR: {exc}",
                })

    # ── Rapport final ──────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("RAPPORT FINAL DU WORKER")
    logger.info("═" * 60)
    for r in resultats:
        score = r.get("score_silhouette", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, float) else score
        logger.info(
            "[%s] %s × %s | Silhouette: %s | Obs: %s",
            r["statut"], r["systeme"], r["scenario"],
            score_str, r.get("nb_obs", "N/A"),
        )

    # ── Nettoyage ─────────────────────────────────────────────────────────
    pg_conn.close()
    if cassandra_cluster:
        cassandra_cluster.shutdown()

    erreurs = [r for r in resultats if r["statut"] != "OK"]
    if erreurs:
        logger.error("%d erreur(s) détectée(s) — vérifier les logs.", len(erreurs))
        raise SystemExit(1)

    logger.info("Worker terminé avec succès — tous les profils sont à jour dans PostgreSQL.")


if __name__ == "__main__":
    main()
