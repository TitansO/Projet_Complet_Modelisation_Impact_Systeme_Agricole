-- =============================================================================
-- Climat IA — Schéma PostgreSQL (Base Relationnelle ACID)
-- Rôle : Persistance forte pour profils agronomiques finalisés,
--        scénarios RCP validés et métriques de clustering K-Means.
-- Systèmes TFE : BAU (Grandes Cultures) | Polyculture-Élevage | ICLS (Agroécologique)
-- =============================================================================

-- Extension utile pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────────────────────
-- TYPE ÉNUMÉRÉ : Systèmes de production agricole (3 systèmes TFE)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TYPE systeme_production_type AS ENUM (
    'Systeme_Grandes_Cultures',     -- BAU : système Business-as-usual, grandes cultures wallonnes
    'Systeme_Polyculture_Elevage',  -- Vegan : polyculture sans fumier animal
    'Systeme_Agroecologique'        -- ICLS : Intégré Cultures-Élevage, haute résilience
);

CREATE TYPE scenario_rcp_type AS ENUM (
    'RCP_4.5',   -- Scénario climatique modéré (stabilisation ~650 ppm CO2 eq)
    'RCP_8.5'    -- Scénario pessimiste (business-as-usual, ~1370 ppm CO2 eq en 2100)
);

CREATE TYPE impact_rendement_type AS ENUM (
    'HAUSSE',    -- Cluster 0 : tropical humide, +12.5%
    'STABLE',    -- Cluster 2 : tempéré méditerranéen, +2.0%
    'BAISSE',    -- Cluster 3 : semi-aride transitoire, -18.5%
    'CRITIQUE'   -- Cluster 1 : aride Sahel, -38.0%
);

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 1 : profils_agronomiques
-- Stocke les profils de vulnérabilité finalisés, produits par le Worker K-Means
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS profils_agronomiques (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    systeme_production    systeme_production_type NOT NULL,
    scenario_rcp          scenario_rcp_type       NOT NULL,
    cluster_id            SMALLINT                NOT NULL CHECK (cluster_id BETWEEN 0 AND 9),
    profil_climatique     VARCHAR(64)             NOT NULL,
    impact_rendement      impact_rendement_type   NOT NULL,

    -- Variables climatiques du centroïde du cluster
    temperature_c_moy     NUMERIC(5,2) NOT NULL CHECK (temperature_c_moy BETWEEN -20 AND 60),
    pluviometrie_mm_moy   NUMERIC(7,2) NOT NULL CHECK (pluviometrie_mm_moy >= 0),
    indice_secheresse_moy NUMERIC(4,3) NOT NULL CHECK (indice_secheresse_moy BETWEEN 0 AND 1),
    radiation_kwh_moy     NUMERIC(6,2) NOT NULL CHECK (radiation_kwh_moy >= 0),

    -- Score de qualité du clustering
    score_silhouette      NUMERIC(4,3),

    -- Métadonnées
    periode_debut         SMALLINT NOT NULL,
    periode_fin           SMALLINT NOT NULL,
    nb_observations       INTEGER  NOT NULL DEFAULT 0,
    cree_le               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mis_a_jour_le         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unicité métier : un profil par (système, scénario, cluster)
    CONSTRAINT uq_profil_systeme_scenario_cluster
        UNIQUE (systeme_production, scenario_rcp, cluster_id)
);

COMMENT ON TABLE profils_agronomiques IS
    'Profils de vulnérabilité agronomique finalisés par le Worker K-Means. '
    'Source primaire de vérité pour les prédictions de l''API FastAPI.';

CREATE INDEX idx_profils_systeme ON profils_agronomiques(systeme_production);
CREATE INDEX idx_profils_scenario ON profils_agronomiques(scenario_rcp);
CREATE INDEX idx_profils_cluster ON profils_agronomiques(cluster_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 2 : metriques_clustering
-- Métriques de performance des runs K-Means (traçabilité analytique)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS metriques_clustering (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id                UUID NOT NULL DEFAULT uuid_generate_v4(),
    systeme_production    systeme_production_type NOT NULL,
    scenario_rcp          scenario_rcp_type       NOT NULL,
    n_clusters            SMALLINT                NOT NULL,
    inertie_totale        NUMERIC(12,4),
    score_silhouette_moy  NUMERIC(4,3),
    nb_iterations         SMALLINT,
    nb_observations       INTEGER,
    features_utilisees    TEXT[],  -- ['temperature_c','pluviometrie_mm','indice_secheresse','radiation_kwh']
    execute_le            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duree_ms              INTEGER
);

COMMENT ON TABLE metriques_clustering IS
    'Métriques de chaque exécution du pipeline K-Means. '
    'Permet le suivi de la dérive de modèle (model drift) dans le temps.';

CREATE INDEX idx_metriques_run ON metriques_clustering(run_id);
CREATE INDEX idx_metriques_systeme ON metriques_clustering(systeme_production);

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 3 : logs_inference
-- Trace chaque appel /predict de l'API (avec info cache Redis)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS logs_inference (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    systeme_production    systeme_production_type NOT NULL,
    scenario_rcp          scenario_rcp_type,
    temperature_c         NUMERIC(5,2),
    pluviometrie_mm       NUMERIC(7,2),
    indice_secheresse     NUMERIC(4,3),
    radiation_kwh         NUMERIC(6,2),
    cluster_id_predit     SMALLINT,
    impact_predit         impact_rendement_type,
    cache_hit             BOOLEAN NOT NULL DEFAULT FALSE,
    latence_ms            INTEGER,
    demande_le            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE logs_inference IS
    'Journal de chaque appel d''inférence via l''API. '
    'Le champ cache_hit indique si la réponse provenait du cache Redis (TRUE) '
    'ou d''une requête PostgreSQL (FALSE — Cache Miss).';

CREATE INDEX idx_logs_systeme ON logs_inference(systeme_production);
CREATE INDEX idx_logs_date ON logs_inference(demande_le);

-- ─────────────────────────────────────────────────────────────────────────────
-- DONNÉES INITIALES : 4 profils climatiques de référence (centroïdes K-Means)
-- Valeurs issues de l'analyse exploratoire du TFE Delandmeter (2021)
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO profils_agronomiques (
    systeme_production, scenario_rcp, cluster_id, profil_climatique,
    impact_rendement, temperature_c_moy, pluviometrie_mm_moy,
    indice_secheresse_moy, radiation_kwh_moy, score_silhouette,
    periode_debut, periode_fin, nb_observations
) VALUES
-- Système Grandes Cultures (BAU) — Scénario RCP 4.5
('Systeme_Grandes_Cultures', 'RCP_4.5', 0, 'Tropical humide',
 'HAUSSE', 27.0, 1450.0, 0.18, 95.0, 0.72, 2045, 2069, 120),

('Systeme_Grandes_Cultures', 'RCP_4.5', 1, 'Aride chaud (Sahel)',
 'CRITIQUE', 36.0, 280.0, 0.82, 210.0, 0.68, 2045, 2069, 120),

('Systeme_Grandes_Cultures', 'RCP_4.5', 2, 'Tempéré méditerranéen',
 'STABLE', 16.0, 680.0, 0.35, 155.0, 0.61, 2045, 2069, 120),

('Systeme_Grandes_Cultures', 'RCP_4.5', 3, 'Semi-aride transitoire',
 'BAISSE', 29.0, 420.0, 0.58, 175.0, 0.55, 2045, 2069, 120),

-- Système Agroécologique (ICLS) — Scénario RCP 8.5 (haute résilience hydrique)
('Systeme_Agroecologique', 'RCP_8.5', 0, 'Tropical humide',
 'HAUSSE', 29.5, 1380.0, 0.21, 105.0, 0.74, 2075, 2099, 120),

('Systeme_Agroecologique', 'RCP_8.5', 1, 'Aride chaud (Sahel)',
 'BAISSE', 38.5, 195.0, 0.89, 235.0, 0.66, 2075, 2099, 120),

('Systeme_Agroecologique', 'RCP_8.5', 2, 'Tempéré méditerranéen',
 'STABLE', 19.0, 590.0, 0.38, 162.0, 0.63, 2075, 2099, 120),

('Systeme_Agroecologique', 'RCP_8.5', 3, 'Semi-aride transitoire',
 'STABLE', 31.0, 365.0, 0.62, 188.0, 0.58, 2075, 2099, 120)

ON CONFLICT (systeme_production, scenario_rcp, cluster_id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────────
-- TRIGGER : mise à jour automatique de mis_a_jour_le
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_mis_a_jour_le()
RETURNS TRIGGER AS $$
BEGIN
    NEW.mis_a_jour_le = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_profils_updated
    BEFORE UPDATE ON profils_agronomiques
    FOR EACH ROW EXECUTE FUNCTION update_mis_a_jour_le();

-- Confirmation
DO $$
BEGIN
    RAISE NOTICE 'Schéma Climat IA initialisé avec succès — PostgreSQL 16';
    RAISE NOTICE 'Tables créées : profils_agronomiques, metriques_clustering, logs_inference';
END $$;
