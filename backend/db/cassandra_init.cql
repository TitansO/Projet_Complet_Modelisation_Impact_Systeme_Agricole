-- =============================================================================
-- Climat IA — Schéma Apache Cassandra 4.1 (Base NoSQL Séries Temporelles)
-- Rôle : Stockage distribué haute disponibilité (modèle AP, anneau masterless)
--        pour les flux massifs de données climatiques brutes des capteurs.
-- Architecture : Replication Factor 1 (dev) → 3 en production multi-nœuds
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- KEYSPACE : Unité d'isolation Cassandra (équivalent schéma PostgreSQL)
-- NetworkTopologyStrategy préparé pour le datacenter ouest_africain
-- ─────────────────────────────────────────────────────────────────────────────
CREATE KEYSPACE IF NOT EXISTS climat_ia_ks
    WITH REPLICATION = {
        'class'              : 'SimpleStrategy',   -- NetworkTopologyStrategy en prod
        'replication_factor' : 1                   -- RF=3 pour production multi-nœuds
    }
    AND DURABLE_WRITES = true;

USE climat_ia_ks;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE PRINCIPALE : flux_climatiques_bruts
-- Modèle de données optimisé pour les requêtes de séries temporelles.
--
-- CLÉ DE PARTITION : (systeme_production, date_mesure)
--   → Distribue les données par système agricole ET par jour, limitant la taille
--     des partitions et garantissant la localité des données pour les requêtes
--     de type "dernier mois pour le système X".
--
-- CLÉ DE CLUSTERING : (heure_mesure DESC, capteur_id)
--   → Tri décroissant = accès O(1) aux mesures les plus récentes (time-series pattern).
--   → capteur_id en CLÉ de clustering garantit l'unicité sub-milliseconde.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flux_climatiques_bruts (

    -- Clé de partition composite
    systeme_production  TEXT,       -- 'Systeme_Grandes_Cultures' | 'Systeme_Polyculture_Elevage' | 'Systeme_Agroecologique'
    date_mesure         DATE,       -- Partitionnement journalier pour contrôler la taille des partitions

    -- Clé de clustering (ordre temporel décroissant + unicité capteur)
    heure_mesure        TIMESTAMP,  -- Précision milliseconde (horloge capteur IoT)
    capteur_id          TEXT,       -- Identifiant unique du capteur physique déployé

    -- Variables climatiques brutes (non normalisées — raw IoT data)
    temperature_c       FLOAT,      -- Température en °C (plage : -20 à 60°C)
    pluviometrie_mm     FLOAT,      -- Pluviométrie en mm (plage : 0 à 500 mm/j)
    humidite_pct        FLOAT,      -- Humidité relative en % (0-100)
    indice_secheresse   FLOAT,      -- Indice de sécheresse calculé (0=humide, 1=aride)
    radiation_kwh       FLOAT,      -- Radiation solaire en kWh/m² (0-15 kWh/m²/j)
    vitesse_vent_ms     FLOAT,      -- Vitesse du vent en m/s

    -- Métadonnées du capteur
    station_id          TEXT,       -- Station météorologique source
    qualite_donnee      TEXT,       -- 'VALIDE' | 'DOUTEUX' | 'MANQUANT'
    scenario_rcp        TEXT,       -- Contexte RCP pour les données simulées

    PRIMARY KEY (
        (systeme_production, date_mesure),  -- Partition composite
        heure_mesure,                        -- Clustering 1 (DESC pour recency)
        capteur_id                           -- Clustering 2 (unicité)
    )
) WITH CLUSTERING ORDER BY (heure_mesure DESC, capteur_id ASC)
  AND compaction = {
      'class'             : 'TimeWindowCompactionStrategy',
      'compaction_window_unit'  : 'DAYS',
      'compaction_window_size'  : 7
  }
  AND compression = {
      'sstable_compression' : 'LZ4Compressor'
  }
  AND default_time_to_live = 7776000   -- TTL 90 jours pour les données brutes
  AND comment = 'Flux de séries temporelles climatiques brutes. Partition par (système, jour). '
                'TWCS optimisé pour l''insertion en fenêtre glissante de 7 jours.';

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE : agregats_quotidiens
-- Agrégats pré-calculés par capteur et par jour (optimisation des requêtes batch)
-- Pattern Materialized View / Summary table pour le Worker de clustering
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agregats_quotidiens (

    systeme_production  TEXT,
    annee_mois          TEXT,       -- Format 'YYYY-MM' pour partitionnement mensuel
    date_agregat        DATE,
    capteur_id          TEXT,

    -- Statistiques descriptives journalières
    temperature_moy     FLOAT,
    temperature_max     FLOAT,
    temperature_min     FLOAT,
    pluviometrie_totale FLOAT,
    humidite_moy        FLOAT,
    indice_secheresse_moy FLOAT,
    radiation_totale    FLOAT,
    nb_mesures          INT,        -- Nombre de mesures brutes ayant contribué
    calcule_le          TIMESTAMP,

    PRIMARY KEY (
        (systeme_production, annee_mois),
        date_agregat,
        capteur_id
    )
) WITH CLUSTERING ORDER BY (date_agregat DESC, capteur_id ASC)
  AND compaction = {
      'class' : 'LeveledCompactionStrategy'
  }
  AND comment = 'Agrégats journaliers pré-calculés pour alimenter le Worker K-Means. '
                'Évite les full-scan sur flux_climatiques_bruts lors du clustering batch.';

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE : statut_capteurs
-- Table de lookup pour la supervision de la flotte de capteurs IoT
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS statut_capteurs (
    capteur_id          TEXT PRIMARY KEY,
    systeme_production  TEXT,
    station_id          TEXT,
    latitude            FLOAT,
    longitude           FLOAT,
    actif               BOOLEAN,
    derniere_mesure     TIMESTAMP,
    firmware_version    TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- INDEX SECONDAIRES (utiliser avec précaution sur Cassandra — partition scan)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_flux_qualite
    ON flux_climatiques_bruts (qualite_donnee);

CREATE INDEX IF NOT EXISTS idx_flux_scenario
    ON flux_climatiques_bruts (scenario_rcp);

-- Confirmation
-- (Note : en production, exécuter ce script via : cqlsh -f cassandra_init.cql)
