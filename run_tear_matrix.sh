#!/usr/bin/env bash
# run_tear_matrix.sh — run tear_test across SMT / core / die / NUMA-node / socket placements.
#
#   ./run_tear_matrix.sh                  60 s per placement + 5 s per positive control
#   ./run_tear_matrix.sh -t 3600 -c 10    long soak
#   ./run_tear_matrix.sh -n               print topology, plan and commands; run nothing
#
# Options
#   -t SECS   main run length (aligned vmovdqa64 store + load), default 60
#   -c SECS   length of each positive-control run, default 5
#   -b CPU    writer CPU the placements are built around (default: first online CPU)
#   -P LIST   placements to run, comma-separated (default: every one this box has):
#             smt core far die node node-rhome socket socket-rhome multi flush
#   -o FILE   results TSV (default tear_results_<host>_<time>.tsv); full output in FILE.log
#   -x        skip the positive controls
#   -n        dry run
#   -- ARGS   extra arguments for every tear_test run, e.g. -- --store-pause 4
#
# Placements (the writer is always on the base CPU):
#   smt           reader on the base CPU's SMT sibling (shared L1D)
#   core          reader on another core in the same package, die and NUMA node
#   far           reader on the highest-numbered core in that node (likely the far side of the mesh)
#   die           reader on another die_id in the same package and node, if sysfs reports dies
#   node          reader on another NUMA node in the same package (SNC / other compute die)
#   socket        reader on another package (UPI)
#   *-rhome       node / socket again, with the test page homed on the reader's node
#   multi         one reader in each of the locations above at the same time
#   flush         core placement plus a third thread clflushing the line (memory path)
#
# Positive controls, run per placement (they should tear; one that doesn't means low power):
#   split  --offset 32               same zmm instruction, split across two lines
#   ymm2s  --store ymm2 --gap 1      two 32-byte stores with a pause between them
#   ymm2l  --load ymm2 --gap 1       two 32-byte loads with a pause between them
#
# SYSFS_ROOT=/path overrides /sys (for checking the placement logic against a mock tree).
# Exit status: 0 if no main run tore, 2 if one did, 1 on setup errors.

set -euo pipefail

MAIN_SECS=60 CTRL_SECS=5 BASE="" ONLY="" OUT="" CONTROLS=1 DRY=0
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
while getopts "t:c:b:P:o:xnh" opt; do
  case $opt in
    t) MAIN_SECS=$OPTARG ;;
    c) CTRL_SECS=$OPTARG ;;
    b) BASE=$OPTARG ;;
    P) ONLY=$OPTARG ;;
    o) OUT=$OPTARG ;;
    x) CONTROLS=0 ;;
    n) DRY=1 ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
EXTRA=("$@")

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN=${TEAR_TEST:-$HERE/tear_test}
SRC=$HERE/tear_test.cpp
SYS=${SYSFS_ROOT:-/sys}

# ---------------------------------------------------------------- topology
expand() {  # "0-3,8,10-11" -> "0 1 2 3 8 10 11"
  local part a b i out=() parts=()
  IFS=, read -ra parts <<< "$1"
  for part in ${parts[@]+"${parts[@]}"}; do
    part=${part//[[:space:]]/}
    [[ -z $part ]] && continue
    if [[ $part == *-* ]]; then
      a=${part%-*} b=${part#*-}
      for ((i = a; i <= b; i++)); do out+=("$i"); done
    else
      out+=("$part")
    fi
  done
  echo ${out[@]+"${out[@]}"}
}
read_or() { if [[ -r $1 ]]; then cat "$1"; else echo "$2"; fi; }

read -ra ONLINE <<< "$(expand "$(read_or "$SYS/devices/system/cpu/online" 0)")"
declare -A PKG CORE DIE NODE SIB
for c in "${ONLINE[@]}"; do
  t=$SYS/devices/system/cpu/cpu$c/topology
  PKG[$c]=$(read_or "$t/physical_package_id" 0)
  CORE[$c]=$(read_or "$t/core_id" "$c")
  DIE[$c]=$(read_or "$t/die_id" 0)
  SIB[$c]=$(expand "$(read_or "$t/thread_siblings_list" "$c")")
  NODE[$c]=0
done
for nd in "$SYS"/devices/system/node/node[0-9]*; do
  [[ -r $nd/cpulist ]] || continue
  n=${nd##*/node}
  for c in $(expand "$(cat "$nd/cpulist")"); do
    [[ -n ${PKG[$c]+x} ]] && NODE[$c]=$n
  done
done

BASE=${BASE:-${ONLINE[0]}}
[[ -n ${PKG[$BASE]+x} ]] || { echo "error: cpu $BASE is not online" >&2; exit 1; }

is_sib()  { [[ " ${SIB[$BASE]} " == *" $1 "* ]]; }
primary() { local first; read -r first _ <<< "${SIB[$1]}"; [[ $first == "$1" ]]; }

SMT="" CORE1="" FAR="" DIE1="" NODE1="" SOCK=""
for s in ${SIB[$BASE]}; do
  if [[ $s != "$BASE" ]]; then SMT=$s; break; fi
done
for c in "${ONLINE[@]}"; do
  [[ $c == "$BASE" ]] && continue
  is_sib "$c" && continue
  primary "$c" || continue
  if [[ ${PKG[$c]} != "${PKG[$BASE]}" ]]; then
    [[ -z $SOCK ]] && SOCK=$c
  elif [[ ${NODE[$c]} != "${NODE[$BASE]}" ]]; then
    [[ -z $NODE1 ]] && NODE1=$c
  elif [[ ${DIE[$c]} != "${DIE[$BASE]}" ]]; then
    [[ -z $DIE1 ]] && DIE1=$c
  else
    [[ -z $CORE1 ]] && CORE1=$c
    FAR=$c  # last match wins: highest-numbered core in the base CPU's node and die
  fi
done
[[ $FAR == "$CORE1" ]] && FAR=""

PLAN=()  # name|writers|readers|flusher|node
add() { PLAN+=("$1|$2|$3|$4|$5"); }
[[ -n $SMT ]] && add smt "$BASE" "$SMT" - -
[[ -n $CORE1 ]] && add core "$BASE" "$CORE1" - -
[[ -n $FAR ]] && add far "$BASE" "$FAR" - -
[[ -n $DIE1 ]] && add die "$BASE" "$DIE1" - -
if [[ -n $NODE1 ]]; then
  add node "$BASE" "$NODE1" - -
  add node-rhome "$BASE" "$NODE1" - "${NODE[$NODE1]}"
fi
if [[ -n $SOCK ]]; then
  add socket "$BASE" "$SOCK" - -
  add socket-rhome "$BASE" "$SOCK" - "${NODE[$SOCK]}"
fi
MULTI="" nmulti=0
for c in "$SMT" "$CORE1" "$FAR" "$DIE1" "$NODE1" "$SOCK"; do
  if [[ -n $c ]]; then MULTI+="${MULTI:+,}$c"; nmulti=$((nmulti + 1)); fi
done
((nmulti >= 2)) && add multi "$BASE" "$MULTI" - -
FL_READER=${CORE1:-$SMT} FLUSHER=""
for c in "$SOCK" "$NODE1" "$DIE1" "$FAR"; do
  if [[ -n $c && $c != "$FL_READER" ]]; then FLUSHER=$c; break; fi
done
if [[ -z $FLUSHER ]]; then
  for c in "${ONLINE[@]}"; do
    if [[ $c != "$BASE" && $c != "$FL_READER" ]] && ! is_sib "$c"; then FLUSHER=$c; break; fi
  done
fi
[[ -n $FL_READER && -n $FLUSHER ]] && add flush "$BASE" "$FL_READER" "$FLUSHER" -

if [[ -n $ONLY ]]; then
  keep=()
  for p in ${PLAN[@]+"${PLAN[@]}"}; do
    [[ ",$ONLY," == *",${p%%|*},"* ]] && keep+=("$p")
  done
  PLAN=(${keep[@]+"${keep[@]}"})
fi
if ((${#PLAN[@]} == 0)); then
  echo "error: no placements to run (this needs at least two usable CPUs)" >&2
  exit 1
fi

desc() { echo "cpu $1 (pkg ${PKG[$1]}, die ${DIE[$1]}, node ${NODE[$1]}, core ${CORE[$1]})"; }
echo "base     $(desc "$BASE")"
for pair in SMT:smt CORE1:core FAR:far DIE1:die NODE1:node SOCK:socket; do
  v=${pair%%:*}
  if [[ -n ${!v} ]]; then printf '%-8s %s\n' "${pair#*:}" "$(desc "${!v}")"; fi
done
nruns=$((${#PLAN[@]} * (CONTROLS ? 4 : 1)))
total=$(awk -v n=${#PLAN[@]} -v m="$MAIN_SECS" -v c="$CTRL_SECS" -v k=$CONTROLS \
  'BEGIN { printf "%.0f", n * (m + 3 * c * k) }')
echo "plan     ${#PLAN[@]} placements, $nruns runs, about ${total} s"

# ---------------------------------------------------------------- runs
if ((!DRY)); then
  if [[ ! -x $BIN || ( -f $SRC && $SRC -nt $BIN ) ]]; then
    [[ -f $SRC ]] || { echo "error: $BIN not found and no $SRC to build it from" >&2; exit 1; }
    echo "building $BIN"
    "${CXX:-g++}" -O2 -std=c++20 -pthread -o "$BIN" "$SRC"
  fi
  OUT=${OUT:-tear_results_$(hostname -s)_$(date +%Y%m%d-%H%M%S).tsv}
  LOG=${OUT%.tsv}.log
  printf 'placement\tmode\twriters\treaders\tflusher\tnode\tsecs\tstores\tloads\ttransitions\ttears\tboundaries\tverdict\n' > "$OUT"
  : > "$LOG"
fi

field() {  # field RESULT_LINE key
  local v
  v=$(sed -n "s/.* $2=\([^ ]*\).*/\1/p" <<< "$1")
  echo "${v:--}"
}

run_one() {  # name mode writers readers flusher node secs [tear_test args...]
  local name=$1 mode=$2 w=$3 r=$4 f=$5 node=$6 secs=$7
  shift 7
  local cmd=("$BIN" --writers "$w" --readers "$r" --seconds "$secs")
  [[ $f != - ]] && cmd+=(--flusher "$f")
  [[ $node != - ]] && cmd+=(--node "$node")
  cmd+=("$@" ${EXTRA[@]+"${EXTRA[@]}"})
  if ((DRY)); then
    printf '  %-13s %-6s %s\n' "$name" "$mode" "${cmd[*]}"
    return 0
  fi
  printf '%-13s %-6s ' "$name" "$mode"
  local out rc=0 res tears verdict
  out=$("${cmd[@]}" 2>&1) || rc=$?
  { echo "### $name $mode: ${cmd[*]}"; echo "$out"; echo; } >> "$LOG"
  res=$(grep '^RESULT ' <<< "$out" | tail -n 1 || true)
  tears=$(field "$res" tears)
  if [[ -z $res || $rc == 1 ]]; then
    verdict=ERROR
  elif [[ $mode == main ]]; then
    if [[ $tears == 0 ]]; then verdict=no-tear; else verdict=TORN; fi
  else
    if [[ $tears != 0 ]]; then verdict=tore-ok; else verdict=NO-TEAR-weak; fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$mode" "$w" "$r" "$f" \
    "$(field "$res" node)" "$(field "$res" secs)" "$(field "$res" stores)" "$(field "$res" loads)" \
    "$(field "$res" transitions)" "$tears" "$(field "$res" boundaries)" "$verdict" >> "$OUT"
  echo "$verdict  transitions=$(field "$res" transitions) tears=$tears boundaries=$(field "$res" boundaries)"
  if [[ $verdict == ERROR ]]; then grep -m 3 -i 'error\|warning' <<< "$out" | sed 's/^/    /' || true; fi
}

for p in "${PLAN[@]}"; do
  IFS='|' read -r name w r f node <<< "$p"
  run_one "$name" main "$w" "$r" "$f" "$node" "$MAIN_SECS"
  if ((CONTROLS)); then
    run_one "$name" split "$w" "$r" "$f" "$node" "$CTRL_SECS" --offset 32
    run_one "$name" ymm2s "$w" "$r" "$f" "$node" "$CTRL_SECS" --store ymm2 --gap 1
    run_one "$name" ymm2l "$w" "$r" "$f" "$node" "$CTRL_SECS" --load ymm2 --gap 1
  fi
done
((DRY)) && exit 0

echo
if command -v column > /dev/null; then column -t -s $'\t' "$OUT"; else cat "$OUT"; fi
count() { awk -F'\t' -v m="$1" -v v="$2" 'NR > 1 && ($2 == "main") == (m == "main") && $13 == v' "$OUT" | wc -l; }
torn=$(count main TORN) weak=$(count ctrl NO-TEAR-weak) err=$(awk -F'\t' 'NR > 1 && $13 == "ERROR"' "$OUT" | wc -l)
echo
echo "main runs torn: $torn   controls that did not tear: $weak   errors: $err"
echo "results: $OUT   full output: $LOG"
((err == 0)) || exit 1
((torn == 0)) || exit 2
