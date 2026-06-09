#!/usr/bin/env bash
# =============================================================================
# Climat IA — Script de Déploiement & Chaos Engineering
# Architecture Polyglotte : PostgreSQL · Apache Cassandra · Redis · FastAPI
# =============================================================================
set -euo pipefail

RESET='\033[0m'; BOLD='\033[1m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'

log_info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error() { echo -e "${RED}[ERROR]${RESET} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Vérification des prérequis
# ─────────────────────────────────────────────────────────────────────────────
check_prerequisites() {
    log_step "Vérification des prérequis"
    command -v docker  >/dev/null 2>&1 || { log_error "Docker non installé"; exit 1; }
    command -v docker  >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 \
        || { log_error "Docker Compose v2 non disponible"; exit 1; }

    [[ -f ".env" ]] || { log_error ".env absent. Exécuter : cp .env.example .env && éditez les secrets"; exit 1; }

    # RAM disponible (>= 4 Go recommandés pour Cassandra)
    AVAILABLE_RAM_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
    if [[ "$AVAILABLE_RAM_MB" -lt 3000 ]]; then
        log_warn "RAM disponible : ${AVAILABLE_RAM_MB}MB (< 3Go recommandés pour Cassandra)"
    fi
    log_info "Prérequis validés."
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Déploiement complet de la stack
# ─────────────────────────────────────────────────────────────────────────────
deploy_full_stack() {
    log_step "Déploiement de la Stack Polyglotte Climat IA"

    log_info "Validation de la syntaxe docker-compose.yml…"
    docker compose config --quiet && log_info "Syntaxe valide ✓"

    log_info "Build des images personnalisées (API + Worker)…"
    DOCKER_BUILDKIT=1 docker compose build --no-cache api_climat worker_clustering

    log_info "Démarrage de l'infrastructure (Postgres → Cassandra → Redis)…"
    docker compose up --detach --wait \
        base_donnees_postgres nosql_cassandra cache_redis

    log_info "Attente de la disponibilité de Cassandra (healthcheck nodetool status)…"
    RETRIES=0; MAX_RETRIES=30
    until docker compose exec nosql_cassandra nodetool status 2>/dev/null | grep -q "UN"; do
        RETRIES=$((RETRIES+1))
        [[ "$RETRIES" -ge "$MAX_RETRIES" ]] && { log_error "Cassandra non disponible après ${MAX_RETRIES} essais."; exit 1; }
        log_info "Cassandra démarrage... ($RETRIES/$MAX_RETRIES)"
        sleep 10
    done
    log_info "Cassandra opérationnel ✓"

    log_info "Initialisation du schéma Cassandra…"
    docker compose exec nosql_cassandra cqlsh -f /docker-entrypoint-initdb.d/cassandra_init.cql \
        2>/dev/null || log_warn "Schéma Cassandra déjà initialisé (idempotent)"

    log_info "Démarrage de l'API et du reverse proxy…"
    docker compose up --detach --wait api_climat reverse_proxy

    log_info "Stack déployée avec succès ✓"
    print_endpoints
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Exécution du Worker de Clustering
# ─────────────────────────────────────────────────────────────────────────────
run_worker() {
    log_step "Exécution du Worker K-Means (Cassandra → PostgreSQL)"
    log_info "Lancement du worker_clustering…"
    docker compose run --rm worker_clustering
    log_info "Worker terminé. Profils disponibles dans PostgreSQL ✓"
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Test de l'API
# ─────────────────────────────────────────────────────────────────────────────
test_api() {
    log_step "Tests de l'API FastAPI"
    API_URL="http://localhost/api"

    log_info "Health check…"
    HEALTH=$(curl -sf "${API_URL}/health" || echo '{"status":"unreachable"}')
    echo "  → $HEALTH"

    log_info "Test /predict — Système Grandes Cultures / RCP 4.5 (Cache MISS attendu)…"
    PREDICT=$(curl -sf -X POST "${API_URL}/predict" \
        -H "Content-Type: application/json" \
        -d '{"systeme_production":"Systeme_Grandes_Cultures","scenario_rcp":"RCP_4.5",
             "temperature_c":28.0,"pluviometrie_mm":650,"indice_secheresse":0.45,"radiation_kwh":160}' \
        || echo '{"error":"API non disponible"}')
    echo "  → $PREDICT"

    log_info "Test /predict — même requête (Cache HIT Redis attendu)…"
    PREDICT2=$(curl -sf -X POST "${API_URL}/predict" \
        -H "Content-Type: application/json" \
        -d '{"systeme_production":"Systeme_Grandes_Cultures","scenario_rcp":"RCP_4.5",
             "temperature_c":28.0,"pluviometrie_mm":650,"indice_secheresse":0.45,"radiation_kwh":160}' \
        || echo '{}')
    CACHE_HIT=$(echo "$PREDICT2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cache_hit','?'))" 2>/dev/null || echo "?")
    log_info "cache_hit = $CACHE_HIT (attendu : true)"

    log_info "Test /predict — Système Agroécologique / RCP 8.5…"
    curl -sf -X POST "${API_URL}/predict" \
        -H "Content-Type: application/json" \
        -d '{"systeme_production":"Systeme_Agroecologique","scenario_rcp":"RCP_8.5",
             "temperature_c":22.0,"pluviometrie_mm":900,"indice_secheresse":0.20,"radiation_kwh":145}' \
        | python3 -m json.tool 2>/dev/null || log_warn "Formater manuellement la réponse JSON"
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Chaos Engineering — Crash-test Redis (Démo résilience cache)
# ─────────────────────────────────────────────────────────────────────────────
chaos_test_redis() {
    log_step "Chaos Engineering — Panne simulée du Cache Redis"
    log_warn "Arrêt du service cache_redis…"
    docker compose stop cache_redis

    log_info "Test API sans Redis (Cache Miss forcé → fallback PostgreSQL)…"
    RESPONSE=$(curl -sf -X POST "http://localhost/api/predict" \
        -H "Content-Type: application/json" \
        -d '{"systeme_production":"Systeme_Polyculture_Elevage","scenario_rcp":"RCP_4.5",
             "temperature_c":20.0,"pluviometrie_mm":800,"indice_secheresse":0.30,"radiation_kwh":150}' \
        || echo '{"error":"API indisponible"}')
    echo "  → $RESPONSE"
    # L'API doit répondre même sans Redis (la panne Redis est non bloquante)

    log_info "Résurrection du cache Redis…"
    docker compose up --detach --wait cache_redis
    log_info "Redis restauré ✓ — l'API reprend le cache normalement"
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Chaos Engineering — Crash-test PostgreSQL (Démo persistance volume)
# ─────────────────────────────────────────────────────────────────────────────
chaos_test_postgres() {
    log_step "Chaos Engineering — Panne simulée de PostgreSQL (Test Volume)"

    log_info "Phase A — Comptage initial des profils (avant crash)…"
    BEFORE=$(docker compose exec base_donnees_postgres \
        psql -U "${POSTGRES_USER:-climat_user}" -d "${POSTGRES_DB:-climat_ia_db}" \
        -t -c "SELECT COUNT(*) FROM profils_agronomiques;" 2>/dev/null | tr -d ' ')
    log_info "Profils avant crash : ${BEFORE}"

    log_warn "Phase B — Destruction du conteneur PostgreSQL (volume préservé)…"
    docker compose stop base_donnees_postgres
    docker compose rm --force base_donnees_postgres

    log_info "Phase C — Résurrection du nœud…"
    docker compose up --detach --wait base_donnees_postgres

    log_info "Phase D — Vérification de l'intégrité post-crash…"
    sleep 5  # Attente stabilisation
    AFTER=$(docker compose exec base_donnees_postgres \
        psql -U "${POSTGRES_USER:-climat_user}" -d "${POSTGRES_DB:-climat_ia_db}" \
        -t -c "SELECT COUNT(*) FROM profils_agronomiques;" 2>/dev/null | tr -d ' ')
    log_info "Profils après crash : ${AFTER}"

    if [[ "$BEFORE" == "$AFTER" ]]; then
        log_info "✅ INTÉGRITÉ CONFIRMÉE — Le volume postgres_data_vol a préservé toutes les données."
    else
        log_error "❌ INCOHÉRENCE détectée : avant=$BEFORE / après=$AFTER"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Inspection de l'état du cluster
# ─────────────────────────────────────────────────────────────────────────────
inspect_stack() {
    log_step "Inspection de l'État de la Stack"

    log_info "Conteneurs actifs :"
    docker compose ps

    log_info "Nœud Cassandra (nodetool status) :"
    docker compose exec nosql_cassandra nodetool status 2>/dev/null || log_warn "Cassandra non disponible"

    log_info "Cache Redis — infos mémoire :"
    docker compose exec cache_redis redis-cli -a "${REDIS_PASSWORD:-RedisClim@t2025!}" \
        info memory 2>/dev/null | grep -E "used_memory_human|maxmemory_human" || true

    log_info "PostgreSQL — tables et comptages :"
    docker compose exec base_donnees_postgres \
        psql -U "${POSTGRES_USER:-climat_user}" -d "${POSTGRES_DB:-climat_ia_db}" -c \
        "SELECT 'profils_agronomiques' AS table, COUNT(*) FROM profils_agronomiques
         UNION ALL SELECT 'logs_inference', COUNT(*) FROM logs_inference
         UNION ALL SELECT 'metriques_clustering', COUNT(*) FROM metriques_clustering;" \
        2>/dev/null || log_warn "PostgreSQL non disponible"
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Arrêt propre
# ─────────────────────────────────────────────────────────────────────────────
stop_stack() {
    log_step "Arrêt Propre de la Stack"
    docker compose down
    log_info "Stack arrêtée. Volumes préservés."
    log_warn "Pour supprimer les volumes : docker compose down -v"
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire — Affichage des endpoints
# ─────────────────────────────────────────────────────────────────────────────
print_endpoints() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║         ENDPOINTS CLIMAT IA                      ║${RESET}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════╣${RESET}"
    echo -e "${BOLD}║${RESET} Interface    : http://localhost/app/              ${BOLD}║${RESET}"
    echo -e "${BOLD}║${RESET} API Docs     : http://localhost/api/docs          ${BOLD}║${RESET}"
    echo -e "${BOLD}║${RESET} Health check : http://localhost/api/health        ${BOLD}║${RESET}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Menu principal
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    deploy)     check_prerequisites && deploy_full_stack ;;
    worker)     run_worker ;;
    test)       test_api ;;
    chaos-redis)   chaos_test_redis ;;
    chaos-postgres) chaos_test_postgres ;;
    inspect)    inspect_stack ;;
    stop)       stop_stack ;;
    all)
        check_prerequisites
        deploy_full_stack
        run_worker
        test_api
        ;;
    help|*)
        echo -e "${BOLD}Usage : $0 <commande>${RESET}"
        echo ""
        echo "  deploy          Déploie toute la stack Docker"
        echo "  worker          Exécute le pipeline K-Means (Cassandra → PostgreSQL)"
        echo "  test            Teste l'API FastAPI (avec vérification cache Redis)"
        echo "  chaos-redis     Simule une panne Redis (démo résilience API)"
        echo "  chaos-postgres  Simule une panne PostgreSQL (démo persistance volume)"
        echo "  inspect         Inspecte l'état de la stack"
        echo "  stop            Arrête la stack (volumes préservés)"
        echo "  all             deploy + worker + test (démarrage complet)"
        ;;
esac
