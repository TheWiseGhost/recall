#!/usr/bin/env bash
# Developer conveniences. Run `./scripts/dev.sh help` for the list.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DB_CONTAINER=${DB_CONTAINER:-recall-postgres}

usage() {
  cat <<'USAGE'
Usage: ./scripts/dev.sh <command>

  setup       create the venv, install dev extras, start postgres, create the test db
  db          start postgres and wait for it to accept connections
  testdb      create the recall_test database (idempotent)
  check       ruff check + ruff format --check + mypy
  test        unit tests
  test-all    unit + integration tests
  fmt         ruff format + ruff check --fix
  reset       destroy the postgres volume and recreate the schema
  help        this message
USAGE
}

wait_for_db() {
  for _ in $(seq 1 60); do
    if docker exec "$DB_CONTAINER" pg_isready -U recall -d recall >/dev/null 2>&1; then
      echo "postgres ready"
      return 0
    fi
    sleep 1
  done
  echo "postgres did not become ready in time" >&2
  return 1
}

testdb() {
  docker exec "$DB_CONTAINER" psql -U recall -d postgres \
    -c "CREATE DATABASE recall_test OWNER recall;" 2>/dev/null || true
  echo "recall_test ready"
}

case "${1:-help}" in
  setup)
    uv venv --python 3.12
    uv pip install -e ".[dev,pdf]"
    docker compose up -d postgres
    wait_for_db
    testdb
    ;;
  db)      docker compose up -d postgres && wait_for_db ;;
  testdb)  testdb ;;
  check)
    "$PY" -m ruff check .
    "$PY" -m ruff format --check .
    "$PY" -m mypy
    ;;
  test)     "$PY" -m pytest tests/unit -q ;;
  test-all) "$PY" -m pytest -q ;;
  fmt)
    "$PY" -m ruff format .
    "$PY" -m ruff check --fix .
    ;;
  reset)
    docker compose down -v
    docker compose up -d postgres
    wait_for_db
    testdb
    "$PY" -m recall.cli.main migrate
    ;;
  help|*) usage ;;
esac
