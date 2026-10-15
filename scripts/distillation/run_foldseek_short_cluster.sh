#!/usr/bin/env bash
#
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail
IFS=$'\n\t'
umask 002

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_THREADS=128
readonly DEFAULT_PILOT_SIZE=10000

usage() {
    cat <<'EOF'
Run resumable Foldseek clustering for the <=200-aa distillation manifest.

The input manifest must have no header and exactly one relative CIF path per
line:

    root_id/cif_relpath

ROOTS_DIR must contain root_id symlinks.  createdb is deliberately executed
with ROOTS_DIR as its current directory so Foldseek resolves every manifest
entry through those symlinks.

Usage:
  run_foldseek_short_cluster.sh \
    --paths-tsv /mnt/.../output/paths.tsv \
    --roots-dir /mnt/.../output/roots \
    --output-dir /mnt/.../foldseek_short_le200 \
    --foldseek-bin /mnt/.../foldseek \
    [--mode pilot|full] [--pilot-size N] [--threads N] \
    [--tmscore-threshold FLOAT] [--coverage FLOAT] \
    [--work-dir /dev/shm/...] [--min-free-gb N] [--name NAME]

Required:
  --paths-tsv PATH       Generated relative paths.tsv (one path per line).
  --roots-dir PATH       output/roots directory containing root symlinks.
  --output-dir PATH      Persistent DB, result, provenance, and log directory.
  --foldseek-bin PATH    Absolute path to the pinned Foldseek executable.

Options:
  --mode MODE            pilot or full (default: pilot).
  --pilot-size N         First N manifest rows used by pilot (default: 10000).
  --threads N            Foldseek threads (default: 128).
  --tmscore-threshold F  Exact member-normalized TM-score threshold
                         (default: 0.8).
  --coverage F           Bidirectional coverage threshold (default: 0.9).
  --work-dir PATH        Working DB/tmp directory.  Default is a run-specific
                         directory under /dev/shm.
  --min-free-gb N        Required free space in work filesystem.  Defaults:
                         pilot=16 GiB, full=256 GiB.
  --name NAME            Safe output prefix (default: short_le200).
  -h, --help             Show this help.

Fixed production clustering semantics:
  --alignment-type 2
  --tmscore-threshold FLOAT
  --tmscore-threshold-mode 2
  --exact-tmscore 1
  -c FLOAT --cov-mode 0
  --cluster-mode 0
  --cluster-reassign 1

This is the fast local 3Di+amino-acid alignment profile with exact TM-score
rescoring normalized by member length.  It is intentionally NOT global
TMalign.  The Foldseek temporary directory is retained for diagnosis/resume.

Safety:
  All manifests, roots, output, and work paths must be below /mnt or /dev/shm.
  The runner refuses /tmp and any path on the root filesystem.  Use a distinct
  output directory for pilot and full runs.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

info() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

canonical_path() {
    realpath -m -- "$1"
}

require_safe_storage_path() {
    local label="$1"
    local path="$2"
    case "$path" in
        /mnt/* | /dev/shm/*)
            ;;
        *)
            die "$label must be below /mnt or /dev/shm; refused: $path"
            ;;
    esac
    case "$path" in
        /tmp | /tmp/*)
            die "$label must never use /tmp: $path"
            ;;
    esac
}

require_positive_integer() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer, got: $value"
}

require_nonnegative_integer() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$label must be a non-negative integer, got: $value"
}

require_unit_interval_decimal() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] ||
        die "$label must be a decimal between 0 and 1, got: $value"
}

prefix_exists() {
    local prefix="$1"
    [[ -f "${prefix}.dbtype" ]]
}

copy_prefix() {
    local source_prefix="$1"
    local destination_dir="$2"
    local stage_name="$3"
    local source_dir source_base stage_dir
    local -a artifacts

    source_dir="$(dirname "$source_prefix")"
    source_base="$(basename "$source_prefix")"
    shopt -s nullglob
    artifacts=("$source_dir/$source_base"*)
    shopt -u nullglob
    ((${#artifacts[@]} > 0)) || die "No artifacts found for prefix: $source_prefix"
    [[ -f "${source_prefix}.dbtype" ]] || die "Missing Foldseek dbtype artifact: ${source_prefix}.dbtype"

    stage_dir="$destination_dir/.publish-${stage_name}.$$"
    mkdir -p "$stage_dir"
    cp -a -- "${artifacts[@]}" "$stage_dir/"
    shopt -s nullglob
    artifacts=("$stage_dir"/*)
    shopt -u nullglob
    ((${#artifacts[@]} > 0)) || die "Empty publish staging directory: $stage_dir"
    mv -f -- "${artifacts[@]}" "$destination_dir/"
    rmdir -- "$stage_dir"
}

write_done_marker() {
    local marker="$1"
    local stage="$2"
    {
        printf 'stage=%q\n' "$stage"
        printf 'completed_utc=%q\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        printf 'config_sha256=%q\n' "$CONFIG_SHA256"
    } >"${marker}.new.$$"
    mv -f -- "${marker}.new.$$" "$marker"
}

run_foldseek_stage() {
    local stage="$1"
    shift
    local log_path="$LOG_DIR/${stage}.log"
    {
        printf '\n[%s] cwd=%q\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$ROOTS_DIR"
        printf '[command]'
        printf ' %q' "$FOLDSEEK_BIN" "$@"
        printf '\n'
    } | tee -a "$log_path"
    (
        cd "$ROOTS_DIR"
        "$FOLDSEEK_BIN" "$@"
    ) 2>&1 | tee -a "$log_path"
}

MODE="pilot"
PILOT_SIZE="$DEFAULT_PILOT_SIZE"
THREADS="$DEFAULT_THREADS"
TMSCORE_THRESHOLD="0.8"
COVERAGE="0.9"
MIN_FREE_GB=""
NAME="short_le200"
PATHS_TSV=""
ROOTS_DIR=""
OUTPUT_DIR=""
FOLDSEEK_BIN=""
WORK_DIR=""

while (($#)); do
    case "$1" in
        --paths-tsv)
            (($# >= 2)) || die "--paths-tsv requires a value"
            PATHS_TSV="$2"
            shift 2
            ;;
        --roots-dir)
            (($# >= 2)) || die "--roots-dir requires a value"
            ROOTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir requires a value"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --foldseek-bin)
            (($# >= 2)) || die "--foldseek-bin requires a value"
            FOLDSEEK_BIN="$2"
            shift 2
            ;;
        --mode)
            (($# >= 2)) || die "--mode requires a value"
            MODE="$2"
            shift 2
            ;;
        --pilot-size)
            (($# >= 2)) || die "--pilot-size requires a value"
            PILOT_SIZE="$2"
            shift 2
            ;;
        --threads)
            (($# >= 2)) || die "--threads requires a value"
            THREADS="$2"
            shift 2
            ;;
        --tmscore-threshold)
            (($# >= 2)) || die "--tmscore-threshold requires a value"
            TMSCORE_THRESHOLD="$2"
            shift 2
            ;;
        --coverage)
            (($# >= 2)) || die "--coverage requires a value"
            COVERAGE="$2"
            shift 2
            ;;
        --work-dir)
            (($# >= 2)) || die "--work-dir requires a value"
            WORK_DIR="$2"
            shift 2
            ;;
        --min-free-gb)
            (($# >= 2)) || die "--min-free-gb requires a value"
            MIN_FREE_GB="$2"
            shift 2
            ;;
        --name)
            (($# >= 2)) || die "--name requires a value"
            NAME="$2"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$PATHS_TSV" ]] || die "--paths-tsv is required"
[[ -n "$ROOTS_DIR" ]] || die "--roots-dir is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"
[[ -n "$FOLDSEEK_BIN" ]] || die "--foldseek-bin is required"
[[ "$MODE" == "pilot" || "$MODE" == "full" ]] || die "--mode must be pilot or full"
require_positive_integer "--pilot-size" "$PILOT_SIZE"
require_positive_integer "--threads" "$THREADS"
require_unit_interval_decimal "--tmscore-threshold" "$TMSCORE_THRESHOLD"
require_unit_interval_decimal "--coverage" "$COVERAGE"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "Unsafe --name: $NAME"
if [[ -z "$MIN_FREE_GB" ]]; then
    if [[ "$MODE" == "pilot" ]]; then
        MIN_FREE_GB=16
    else
        MIN_FREE_GB=256
    fi
fi
require_nonnegative_integer "--min-free-gb" "$MIN_FREE_GB"

[[ "$FOLDSEEK_BIN" == /* ]] || die "--foldseek-bin must be an absolute path"
PATHS_TSV="$(canonical_path "$PATHS_TSV")"
ROOTS_DIR="$(canonical_path "$ROOTS_DIR")"
OUTPUT_DIR="$(canonical_path "$OUTPUT_DIR")"
FOLDSEEK_BIN="$(realpath -- "$FOLDSEEK_BIN")"
require_safe_storage_path "--paths-tsv" "$PATHS_TSV"
require_safe_storage_path "--roots-dir" "$ROOTS_DIR"
require_safe_storage_path "--output-dir" "$OUTPUT_DIR"
[[ -f "$PATHS_TSV" && -r "$PATHS_TSV" ]] || die "Unreadable paths TSV: $PATHS_TSV"
[[ -s "$PATHS_TSV" ]] || die "Empty paths TSV: $PATHS_TSV"
[[ -d "$ROOTS_DIR" ]] || die "Missing roots directory: $ROOTS_DIR"
[[ -x "$FOLDSEEK_BIN" && -f "$FOLDSEEK_BIN" ]] || die "Foldseek binary is not an executable file: $FOLDSEEK_BIN"

if [[ -z "$WORK_DIR" ]]; then
    output_key="$(printf '%s' "$OUTPUT_DIR" | sha256sum | cut -c1-12)"
    WORK_DIR="/dev/shm/protenix-foldseek-${UID}-${NAME}-${MODE}-${output_key}"
fi
WORK_DIR="$(canonical_path "$WORK_DIR")"
require_safe_storage_path "--work-dir" "$WORK_DIR"
[[ "$WORK_DIR" != "$OUTPUT_DIR" ]] || die "--work-dir and --output-dir must differ"

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.run.lock"
flock -n 9 || die "Another runner holds the lock: $OUTPUT_DIR/.run.lock"

readonly DB_DIR="$OUTPUT_DIR/db"
readonly RESULT_DIR="$OUTPUT_DIR/result"
readonly PROVENANCE_DIR="$OUTPUT_DIR/provenance"
readonly LOG_DIR="$OUTPUT_DIR/logs"
readonly MARKER_DIR="$PROVENANCE_DIR/stages"
readonly WORK_DB_DIR="$WORK_DIR/db"
readonly WORK_TMP_DIR="$WORK_DIR/tmp"
mkdir -p "$DB_DIR" "$RESULT_DIR" "$PROVENANCE_DIR" "$LOG_DIR" "$MARKER_DIR"
mkdir -p "$DB_DIR/input" "$DB_DIR/cluster"
mkdir -p "$WORK_DB_DIR/input" "$WORK_DB_DIR/cluster" "$WORK_TMP_DIR"

available_bytes="$(df -B1 --output=avail "$WORK_DIR" | tail -n 1 | tr -d '[:space:]')"
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die "Cannot determine free bytes for work directory: $WORK_DIR"
required_bytes=$((MIN_FREE_GB * 1024 * 1024 * 1024))
if ((available_bytes < required_bytes)); then
    die "Insufficient work space: available=${available_bytes} bytes, required=${required_bytes} bytes"
fi

readonly FOLDSEEK_SHA256="$(sha256sum "$FOLDSEEK_BIN" | awk '{print $1}')"
readonly MANIFEST_SHA256="$(sha256sum "$PATHS_TSV" | awk '{print $1}')"
readonly MANIFEST_SIZE_BYTES="$(stat -c '%s' "$PATHS_TSV")"
readonly MANIFEST_MTIME_NS="$(stat -c '%Y' "$PATHS_TSV")000000000"

version_tmp="$PROVENANCE_DIR/.foldseek.version.$$"
"$FOLDSEEK_BIN" version >"$version_tmp" 2>&1 || die "Pinned Foldseek binary failed its version command"
if [[ -f "$PROVENANCE_DIR/foldseek.version.txt" ]]; then
    cmp -s "$version_tmp" "$PROVENANCE_DIR/foldseek.version.txt" ||
        die "Foldseek version output changed for an existing run"
    rm -f -- "$version_tmp"
else
    mv -- "$version_tmp" "$PROVENANCE_DIR/foldseek.version.txt"
fi

ACTIVE_MANIFEST="$PATHS_TSV"
if [[ "$MODE" == "pilot" ]]; then
    ACTIVE_MANIFEST="$PROVENANCE_DIR/pilot.paths.tsv"
    pilot_tmp="${ACTIVE_MANIFEST}.new.$$"
    head -n "$PILOT_SIZE" "$PATHS_TSV" >"$pilot_tmp"
    [[ -s "$pilot_tmp" ]] || die "Pilot selection produced an empty manifest"
    if [[ -f "$ACTIVE_MANIFEST" ]]; then
        cmp -s "$pilot_tmp" "$ACTIVE_MANIFEST" ||
            die "Existing pilot manifest differs; use a new output directory"
        rm -f -- "$pilot_tmp"
    else
        mv -- "$pilot_tmp" "$ACTIVE_MANIFEST"
    fi
fi
readonly ACTIVE_MANIFEST
readonly ACTIVE_MANIFEST_SHA256="$(sha256sum "$ACTIVE_MANIFEST" | awk '{print $1}')"
readonly ACTIVE_ROWS="$(wc -l <"$ACTIVE_MANIFEST" | tr -d '[:space:]')"
((ACTIVE_ROWS > 0)) || die "Active manifest has no rows"

config_tmp="$PROVENANCE_DIR/.run-config.$$"
{
    printf 'schema_version=%q\n' "1"
    printf 'runner=%q\n' "$SCRIPT_NAME"
    printf 'mode=%q\n' "$MODE"
    printf 'name=%q\n' "$NAME"
    printf 'threads=%q\n' "$THREADS"
    printf 'pilot_size=%q\n' "$PILOT_SIZE"
    printf 'paths_tsv=%q\n' "$PATHS_TSV"
    printf 'paths_tsv_sha256=%q\n' "$MANIFEST_SHA256"
    printf 'paths_tsv_size_bytes=%q\n' "$MANIFEST_SIZE_BYTES"
    printf 'paths_tsv_mtime_ns=%q\n' "$MANIFEST_MTIME_NS"
    printf 'active_manifest=%q\n' "$ACTIVE_MANIFEST"
    printf 'active_manifest_sha256=%q\n' "$ACTIVE_MANIFEST_SHA256"
    printf 'active_rows=%q\n' "$ACTIVE_ROWS"
    printf 'roots_dir=%q\n' "$ROOTS_DIR"
    printf 'foldseek_bin=%q\n' "$FOLDSEEK_BIN"
    printf 'foldseek_sha256=%q\n' "$FOLDSEEK_SHA256"
    printf 'alignment_type=%q\n' "2"
    printf 'tmscore_threshold=%q\n' "$TMSCORE_THRESHOLD"
    printf 'tmscore_threshold_mode=%q\n' "2"
    printf 'exact_tmscore=%q\n' "1"
    printf 'coverage=%q\n' "$COVERAGE"
    printf 'coverage_mode=%q\n' "0"
    printf 'cluster_mode=%q\n' "0"
    printf 'cluster_reassign=%q\n' "1"
} >"$config_tmp"

readonly CONFIG_PATH="$PROVENANCE_DIR/run-config.env"
if [[ -f "$CONFIG_PATH" ]]; then
    cmp -s "$config_tmp" "$CONFIG_PATH" ||
        die "Run configuration changed; use a new output directory"
    rm -f -- "$config_tmp"
else
    mv -- "$config_tmp" "$CONFIG_PATH"
fi
readonly CONFIG_SHA256="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"

readonly WORK_INPUT_DB="$WORK_DB_DIR/input/$NAME"
readonly WORK_CLUSTER_DB="$WORK_DB_DIR/cluster/$NAME"
readonly PERSISTENT_INPUT_DB="$DB_DIR/input/$NAME"
readonly PERSISTENT_CLUSTER_DB="$DB_DIR/cluster/$NAME"
readonly RESULT_TSV="$RESULT_DIR/${NAME}_clusters.tsv"
readonly CREATEDB_DONE="$MARKER_DIR/createdb.done"
readonly CLUSTER_DONE="$MARKER_DIR/cluster.done"
readonly CREATETSV_DONE="$MARKER_DIR/createtsv.done"

if [[ -f "$CREATEDB_DONE" ]]; then
    prefix_exists "$PERSISTENT_INPUT_DB" || die "createdb marker exists but persistent DB is incomplete"
    if ! prefix_exists "$WORK_INPUT_DB"; then
        info "Restaging persistent input DB into work directory"
        copy_prefix "$PERSISTENT_INPUT_DB" "$WORK_DB_DIR/input" "restage-input"
    fi
    info "Resume: createdb already complete"
else
    info "Stage createdb: rows=$ACTIVE_ROWS threads=$THREADS"
    run_foldseek_stage createdb \
        createdb "$ACTIVE_MANIFEST" "$WORK_INPUT_DB" \
        --threads "$THREADS"
    prefix_exists "$WORK_INPUT_DB" || die "createdb did not produce ${WORK_INPUT_DB}.dbtype"
    copy_prefix "$WORK_INPUT_DB" "$DB_DIR/input" "input"
    write_done_marker "$CREATEDB_DONE" "createdb"
fi

if [[ -f "$CLUSTER_DONE" ]]; then
    prefix_exists "$PERSISTENT_CLUSTER_DB" || die "cluster marker exists but persistent cluster DB is incomplete"
    if ! prefix_exists "$WORK_CLUSTER_DB"; then
        info "Restaging persistent cluster DB into work directory"
        copy_prefix "$PERSISTENT_CLUSTER_DB" "$WORK_DB_DIR/cluster" "restage-cluster"
    fi
    info "Resume: cluster already complete"
else
    info "Stage cluster: fast alignment-type=2 TM>=${TMSCORE_THRESHOLD} member-normalized coverage>=${COVERAGE}"
    run_foldseek_stage cluster \
        cluster "$WORK_INPUT_DB" "$WORK_CLUSTER_DB" "$WORK_TMP_DIR" \
        --threads "$THREADS" \
        --alignment-type 2 \
        --tmscore-threshold "$TMSCORE_THRESHOLD" \
        --tmscore-threshold-mode 2 \
        --exact-tmscore 1 \
        -c "$COVERAGE" \
        --cov-mode 0 \
        --cluster-mode 0 \
        --cluster-reassign 1 \
        --remove-tmp-files 0
    prefix_exists "$WORK_CLUSTER_DB" || die "cluster did not produce ${WORK_CLUSTER_DB}.dbtype"
    copy_prefix "$WORK_CLUSTER_DB" "$DB_DIR/cluster" "cluster"
    write_done_marker "$CLUSTER_DONE" "cluster"
fi

if [[ -f "$CREATETSV_DONE" ]]; then
    [[ -s "$RESULT_TSV" ]] || die "createtsv marker exists but result TSV is missing or empty"
    info "Resume: createtsv already complete"
else
    info "Stage createtsv"
    result_tmp="${RESULT_TSV}.new.$$"
    run_foldseek_stage createtsv \
        createtsv "$WORK_INPUT_DB" "$WORK_INPUT_DB" "$WORK_CLUSTER_DB" "$result_tmp" \
        --threads "$THREADS"
    [[ -s "$result_tmp" ]] || die "createtsv produced an empty result: $result_tmp"
    mv -- "$result_tmp" "$RESULT_TSV"
    write_done_marker "$CREATETSV_DONE" "createtsv"
fi

{
    printf 'completed_utc=%q\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'result_tsv=%q\n' "$RESULT_TSV"
    printf 'result_sha256=%q\n' "$(sha256sum "$RESULT_TSV" | awk '{print $1}')"
    printf 'work_dir=%q\n' "$WORK_DIR"
    printf 'tmp_retained=%q\n' "$WORK_TMP_DIR"
} >"$PROVENANCE_DIR/completion.env.new.$$"
mv -f -- "$PROVENANCE_DIR/completion.env.new.$$" "$PROVENANCE_DIR/completion.env"

info "Complete: $RESULT_TSV"
info "Temporary files retained: $WORK_TMP_DIR"
