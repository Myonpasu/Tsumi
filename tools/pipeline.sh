#!/usr/bin/env bash
# Tsumi's training loop: self-play data from the best network, a candidate trained on it, a match between the two.
#
#   tools/pipeline.sh [first_gen] [last_gen]        e.g. tools/pipeline.sh 0 9999 to run until stopped
#
# It starts from nets/best.nnue, with its checkpoint nets/best.nnue.pt beside it to warm-start from.  Each
# generation N: datagen with the best network -> data/genN, calibration, training -> nets/gen(N+1).nnue, a check
# of the candidate against numpy, and a match against the best network, which the candidate replaces unless it
# scores below it.  Safe to interrupt and run again with the same arguments: completed generations and phases are
# skipped (an interrupted datagen carries on from the games it had; an interrupted training or match starts
# again).  Set the variables below in the environment to change them, e.g.
#   THREADS=16 GAMES=50000 tools/pipeline.sh 0 9999
set -euo pipefail
cd "$(dirname "$0")/.."
# Ctrl-C ends the running phase (tee -i keeps its last lines in the log) and then the loop, even when the phase
# caught the signal itself, as the trainer does.
trap 'echo; echo "interrupted: run the same command again to carry on from here" >&2; exit 130' INT

# Data generation
THREADS=${THREADS:-$(nproc)}            # datagen threads
GAMES=${GAMES:-20000}                   # games per generation
NODES=${NODES:-10000}                   # search nodes per move (each search gets 80-120% of it and finishes its iteration)
RANDOM_OPENINGS=${RANDOM_OPENINGS:-50}  # this % of the games start from a random-play position: 8-12 random plies, played
OPENING_MIN_SCORE=${OPENING_MIN_SCORE:-500}    # when a search scores it within these millipawns for either side
OPENING_MAX_SCORE=${OPENING_MAX_SCORE:-2000}   # (unbalanced, so more often decisive, but not decided; 0: no check)
BOOK=${BOOK:-}                          # the other games start from positions drawn from this book - FEN, EPD, or PGN lines
RANDOM_MOVE_PCT=${RANDOM_MOVE_PCT:-40}  # a random move on this % of the plies from RANDOM_MOVE_MINPLY to
RANDOM_MOVE_MINPLY=${RANDOM_MOVE_MINPLY:-0}    # RANDOM_MOVE_MAXPLY - 1 (counted from the opening) ...
RANDOM_MOVE_MAXPLY=${RANDOM_MOVE_MAXPLY:-20}
RANDOM_MULTI_PV=${RANDOM_MULTI_PV:-3}   # ... one of the top RANDOM_MULTI_PV moves of a MultiPV search within
RANDOM_MULTI_PV_DIFF=${RANDOM_MULTI_PV_DIFF:-250}   # RANDOM_MULTI_PV_DIFF internal units (millipawns) of the best
WRITE_MIN_PLY=${WRITE_MIN_PLY:-5}       # positions before this ply of a game are not recorded
# Training
HIDDEN=${HIDDEN:-1024}                  # units per perspective (a multiple of 64; must match the checkpoint resumed)
EPOCHS=${EPOCHS:-6}                     # epochs per generation; the learning rate halves every 3
LR=${LR:-1e-5}                          # learning rate when warm-starting from the best network's checkpoint
LR_SCRATCH=${LR_SCRATCH:-1e-3}          # learning rate when training from scratch (RESUME=0, or no checkpoint)
LAMBDA=${LAMBDA:-0.8}                   # weight of the search score against the game result in the target at the
END_LAMBDA=${END_LAMBDA:-0.5}           #     first epoch, and at the last (interpolated linearly)
RESULT_FILTER=${RESULT_FILTER:-1}       # 1: a position is trained on with the probability the score model gives its
                                        #     result, so results the score makes unlikely mostly drop out; 0: all kept
LOSS_ASYMMETRY=${LOSS_ASYMMETRY:-0.2}   # the loss is weighted 1 + this where the network is more optimistic than the
                                        #     target
ANCHOR_P=${ANCHOR_P:-0.01}              # the win probability at an output of 0 on the curve the output goes through
                                        #     (+1.00 is a 50% win on it): smaller is steeper (train/wdl.py)
KEEP_BEST=${KEEP_BEST:-1}               # 1: each candidate is the epoch with the lowest validation loss; 0: the last
HISTORY=${HISTORY:-10}                  # train on up to this many generations of data, but only those labelled by the
                                        #     network that labelled this generation's (0: all of those)
MAX_POSITIONS=${MAX_POSITIONS:-0}       # at most this many positions, the newest generations first (0: no cap)
PROGRESS_EVERY=${PROGRESS_EVERY:-2000000}   # a row of the training table every this many positions within an epoch
                                        #     (0: only at the end of each epoch)
DEVICE=${DEVICE:-}                      # e.g. cuda or cpu (default: cuda if available)
RESUME=${RESUME:-1}                     # 1: warm-start each candidate from nets/best.nnue.pt; 0: train from scratch
# Acceptance match
MATCH_GAMES=${MATCH_GAMES:-1000}        # games, in pairs of colours from the same opening
MATCH_NODES=${MATCH_NODES:-50000}       # nodes per move ...
MATCH_TC=${MATCH_TC:-}                  # ... or a time control such as 10+0.1 instead
MATCH_BOOK=${MATCH_BOOK:-}              # a book (as BOOK) to draw openings from; without one, every opening is 8
MATCH_RANDOM_SHARE=${MATCH_RANDOM_SHARE:-50}   # random plies from the start position, with one only this % of them
CONCURRENCY=${CONCURRENCY:-$THREADS}   # games played at once (a game's two engines take turns on one CPU)

first=${1:-0}
last=${2:-5}
for v in GAMES NODES; do
    [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive number (got '${!v}')" >&2; exit 1; }
done
for v in HISTORY MAX_POSITIONS PROGRESS_EVERY RANDOM_OPENINGS OPENING_MIN_SCORE OPENING_MAX_SCORE MATCH_RANDOM_SHARE; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a number (got '${!v}')" >&2; exit 1; }
done
for v in RANDOM_OPENINGS MATCH_RANDOM_SHARE; do
    (( ${!v} <= 100 )) || { echo "$v is a percentage, 0-100 (got ${!v})" >&2; exit 1; }
done
for v in BOOK MATCH_BOOK; do
    [[ -z "${!v}" || -f "${!v}" ]] || { echo "$v: ${!v} not found" >&2; exit 1; }
done
# The trainer's Python environment (numpy, torch, python-chess) must be the one python3 finds.
if ! pyerr=$(python3 -c "import numpy, torch, chess" 2>&1); then
    echo "python3 cannot load the trainer's modules: activate the environment that has numpy, torch and python-chess" >&2
    echo "(see requirements.txt). Python said:" >&2
    echo "$pyerr" | tail -n 3 >&2; exit 1
fi
if ! python3 -c "import sys; sys.path.insert(0, 'train'); from wdl import anchor_probability; anchor_probability(sys.argv[1])" "$ANCHOR_P" 2>/dev/null; then
    echo "ANCHOR_P must be a probability between 0 and 1/2, as a decimal or a fraction (got '$ANCHOR_P')" >&2; exit 1
fi
[[ -x ./tsumi ]] || { echo "./tsumi not found: build the engine first (make)" >&2; exit 1; }
mkdir -p data nets logs
make -s featlib   # the trainer's feature library (rebuilt when its sources change)

best="nets/best.nnue"
if [[ ! -f "$best" ]]; then
    echo "$best is missing: the loop needs a network to play its games. Copy a network there (and its .pt" >&2
    echo "checkpoint as $best.pt for the candidates to warm-start from; without one they are trained from scratch)." >&2
    exit 1
fi
loaded=$(./tsumi "setoption name EvalFile value $best" quit)
loaded=${loaded##*$'\n'}   # the answer to the setoption (the engine may have loaded its default network at startup too)
[[ "$loaded" == *"loaded network"* ]] || { echo "$loaded" >&2; exit 1; }

# ---- console layout: a banner per generation, a rule per phase, aligned key/value lines ---------------------------
rule() { local t="--- $1 "; printf '%s%s\n' "$t" "$(printf '%*s' $((100 - ${#t})) '' | tr ' ' '-')"; }
kv() { printf '  %-16s%s\n' "$1" "$2"; }
fmt_dur() { local s=$1; if ((s < 60)); then echo "${s}s"; elif ((s < 3600)); then printf '%dm%02ds' $((s / 60)) $((s % 60)); else printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60)); fi; }
elapsed() { fmt_dur $(( $(date +%s) - $1 )); }
banner() {   # a full-width banner: the text on the left, the time on the right
    local stamp; stamp="$(date '+%Y-%m-%d %H:%M') "
    printf '\n%s\n' "$(printf '=%.0s' $(seq 100))"
    printf '%s%*s%s\n' "$1" $(( 100 - ${#1} - ${#stamp} > 1 ? 100 - ${#1} - ${#stamp} : 1 )) '' "$stamp"
    printf '%s\n' "$(printf '=%.0s' $(seq 100))"
}

printf '%s\n' "$(printf '=%.0s' $(seq 100))"
printf ' tsumi training loop: generations %s to %s%s\n' "$first" "$last" "$([[ "$last" -ge 9999 ]] && echo " (until stopped)")"
printf '%s\n' "$(printf '=%.0s' $(seq 100))"
kv "data"       "$GAMES games per generation at $NODES nodes, $THREADS threads; random moves on $RANDOM_MOVE_PCT% of plies $RANDOM_MOVE_MINPLY-$((RANDOM_MOVE_MAXPLY - 1)) (top $RANDOM_MULTI_PV within $RANDOM_MULTI_PV_DIFF millipawns); recorded from ply $WRITE_MIN_PLY"
openings=""
if [[ "$RANDOM_OPENINGS" -gt 0 ]]; then
    openings="$RANDOM_OPENINGS% random-play positions"
    [[ "$OPENING_MAX_SCORE" -gt 0 ]] && openings+=" within $OPENING_MIN_SCORE-$OPENING_MAX_SCORE millipawns"
fi
[[ "$RANDOM_OPENINGS" -lt 100 ]] && openings="${openings:+$openings, the rest }${BOOK:+drawn from }${BOOK:-the start position}"
kv "openings"   "$openings"
kv "training"   "$HIDDEN units, $EPOCHS epochs, lr $LR warm start / $LR_SCRATCH from scratch, lambda $LAMBDA$([[ "$LAMBDA" != "$END_LAMBDA" ]] && echo " -> $END_LAMBDA"), anchor p $ANCHOR_P$([[ "$KEEP_BEST" == 1 ]] && echo ", the best epoch kept")"
regularisers=""
[[ "$RESULT_FILTER" == 1 ]] && regularisers="results the score model finds unlikely mostly dropped"
[[ "$LOSS_ASYMMETRY" != 0 ]] && regularisers="${regularisers:+$regularisers, }loss weighted 1 + $LOSS_ASYMMETRY where the network is more optimistic than the target"
[[ -n "$regularisers" ]] && kv "" "$regularisers"
kv ""           "history $([[ "$HISTORY" == 0 ]] && echo "all generations labelled by the same network" || echo "up to $HISTORY generations labelled by the same network")$([[ "$MAX_POSITIONS" -gt 0 ]] && echo ", at most $MAX_POSITIONS positions"), $([[ "$RESUME" == 1 ]] && echo "warm-started from the best network's checkpoint$([[ -f "$best.pt" ]] || echo " (none yet: from scratch until a candidate is accepted)")" || echo "from scratch")"
kv "match"      "$MATCH_GAMES games $([[ -n "$MATCH_TC" ]] && echo "at $MATCH_TC" || echo "at $MATCH_NODES nodes"), $CONCURRENCY at a time; a candidate replaces $best unless it scores below it"
kv ""           "openings: $([[ -n "$MATCH_BOOK" ]] && echo "$((100 - MATCH_RANDOM_SHARE))% drawn from $MATCH_BOOK, $MATCH_RANDOM_SHARE% reached by " || echo "")8 random plies from the start position"

# The network that labels a generation's data, by its md5: written to data/genN/source before datagen, and read
# back to decide which earlier generations may join the training set.
source_id() { md5sum "$best" | cut -d' ' -f1; }

# A generation's results from its logs - data, the candidate's validation loss, the match - with the time of each
# phase when given (gen [data_time train_time match_time]).  Missing logs show as "?".
gen_summary() {
    local g=$1 dt=${2:-} tt=${3:-} mt=${4:-} npos games_line val final
    npos=$(grep -E '^  positions ' "logs/datagen_gen$g.log" 2>/dev/null | tail -n 1 | awk '{print $2}' || true)
    games_line=$(grep -E '^  done ' "logs/datagen_gen$g.log" 2>/dev/null | tail -n 1 | sed 's/^  done *//; s/, [0-9.]* plies per game//' || true)
    val=$(grep -E '^  kept +epoch [0-9]+ of [0-9]+: val loss [0-9.]+' "logs/train_gen$g.log" 2>/dev/null | tail -n 1 \
          | sed -E 's/^  kept +(epoch [0-9]+ of [0-9]+): (val loss [0-9.]+).*/\2 (\1)/' || true)
    final=$(awk '/^Games +\|/ {games=$0; sub(/^Games +\| /, "", games)} /^Elo +\|/ {elo=$0; sub(/^Elo +\| /, "", elo)}
                 END {if (elo != "") print games ", Elo " elo}' "logs/match_gen$g.log" 2>/dev/null || true)
    kv "data"     "${npos:-?} positions in data/gen$g/data.bin from ${games_line:-? games}${dt:+  ($(fmt_dur "$dt"))}"
    kv "training" "nets/gen$((g+1)).nnue: ${val:-?}${tt:+  ($(fmt_dur "$tt"))}"
    kv "match"    "gen$((g+1)) vs best: ${final:-?}${mt:+  ($(fmt_dur "$mt"))}"
}
# The phases of a generation that are complete are marked by files written when they end: data/genN/done,
# logs/train_genN.done and logs/decision_genN, the last holding the decision ("accepted" or "rejected", the Elo and
# its error), written once the best network has been replaced if the candidate was accepted.  A match in progress
# is marked by logs/match_genN.running.  Deleting a phase's product redoes it on the next run: the candidate
# (nets/genN+1.nnue) for the training and the match, the match log for the match alone.
decision_file() { echo "logs/decision_gen$1"; }
match_result() { grep -E '^Elo +\| ' "logs/match_gen$1.log" 2>/dev/null | tail -n 1 | awk '{print $3, $5}' || true; }
verdict() { awk "BEGIN{exit !($1 >= 0)}" && echo accepted || echo rejected; }
# The candidate of generation N replaces the best network: the network first and its checkpoint second, each copied
# beside its target and moved into place, so that no file is ever half-written and a run interrupted in between
# can tell (the best network is then the candidate already) and finish the replacement.
promote() {
    local net="nets/gen$(($1 + 1)).nnue"
    cp "$net" "$best.tmp" && mv "$best.tmp" "$best"
    cp "$net.pt" "$best.pt.tmp" && mv "$best.pt.tmp" "$best.pt"
}

# Completed generations are skipped quietly; the latest of a run of them is summarised once, before the next
# generation with work left (or at the end), so that a re-run shows where the loop stands.
skip_first=""; skip_last=""
show_skipped() {
    [[ -n "$skip_last" ]] || return 0
    local verdict elo err
    if [[ "$skip_first" == "$skip_last" ]]; then banner " generation $skip_last: already complete"
    else banner " generations $skip_first-$skip_last: already complete"; fi
    rule "generation $skip_last: summary"
    gen_summary "$skip_last"
    read -r verdict elo err < "$(decision_file "$skip_last")"
    if [[ "$verdict" == accepted ]]; then
        kv "decision" "accepted: nets/gen$((skip_last + 1)).nnue became $best"
    else
        kv "decision" "rejected: $best stayed"
    fi
    skip_first=""; skip_last=""
}

for gen in $(seq "$first" "$last"); do
    net="nets/gen$((gen+1)).nnue"
    decision=$(decision_file "$gen")
    if [[ ! -f "$decision" ]]; then
        read -r elo err <<< "$(match_result "$gen")"
        if [[ -n "$elo" && -f "$net" ]] && cmp -s "$net" "$best"; then
            # Interrupted while the candidate replaced the best network (see promote): finished here.
            promote "$gen"
            echo "accepted $elo $err" > "$decision"
            rm -f "logs/match_gen$gen.running"
        elif [[ -n "$elo" && ! -f "logs/match_gen$gen.running" ]]; then
            # A finished match without a decision file (logs written before this script kept them): decided by the log.
            echo "$(verdict "$elo") $elo $err" > "$decision"
        fi
    fi
    # A decided generation whose candidate or match log was deleted is redone from there - unless the candidate had
    # been accepted: the network it was matched against is gone.
    redo=""
    if [[ -f "$decision" && ( ! -f "$net" || ! -f "logs/match_gen$gen.log" ) ]]; then
        source=$(cat "data/gen$gen/source" 2>/dev/null || true)
        if [[ "$source" != "$(source_id)" ]]; then
            prev=$(md5sum nets/*.nnue 2>/dev/null | awk -v s="$source" '$1 == s {print $2; exit}')
            echo "generation $gen: $([[ -f "$net" ]] && echo "logs/match_gen$gen.log" || echo "$net") is missing, but its candidate was accepted and is now" >&2
            echo "$best. To redo the generation, make the network that generated data/gen$gen the best network again first:" >&2
            if [[ -n "$prev" ]]; then echo "    cp $prev $best && cp $prev.pt $best.pt" >&2
            else echo "    (md5 $source; it is no longer in nets/)" >&2; fi
            exit 1
        fi
        if [[ -f "$net" ]]; then redo="logs/match_gen$gen.log is missing: the match is played again"
        else redo="$net is missing: the candidate is trained and matched again"; rm -f "logs/train_gen$gen.done"; fi
        rm -f "$decision"
    fi
    if [[ -f "data/gen$gen/done" && -f "$decision" ]]; then
        skip_first=${skip_first:-$gen}
        skip_last=$gen
        continue
    fi
    show_skipped
    gen_start=$(date +%s)
    banner " generation $gen: data from $best, candidate $net"
    [[ -z "$redo" ]] || kv "note" "$redo"
    mkdir -p "data/gen$gen"
    # An incomplete generation's data, finished or partial, must come from the current best network.  Data from
    # another network (the best one replaced by hand since) is set aside and generated afresh.
    trained="logs/train_gen$gen.done"
    if [[ -f "data/gen$gen/data.bin" || -f "data/gen$gen/done" ]]; then
        have=$(cat "data/gen$gen/source" 2>/dev/null || echo unknown)
        want=$(source_id)
        if [[ "$have" != "$want" ]]; then
            old="data/gen$gen.old"; n=1
            while [[ -e "$old" ]]; do old="data/gen$gen.old$n"; n=$((n + 1)); done
            kv "note" "data/gen$gen was generated by network $have, but the best network is $want: moved to $old"
            mv "data/gen$gen" "$old"
            mkdir -p "data/gen$gen"
            rm -f "$trained"
        fi
    fi

    rule "data generation"
    phase_start=$(date +%s)
    if [[ ! -f "data/gen$gen/done" ]]; then
        rm -f "$trained"
        source_id > "data/gen$gen/source"
        bookopt=()
        if [[ -n "$BOOK" && "$RANDOM_OPENINGS" -lt 100 ]]; then   # the games that do not start from a random-play position
            if [[ -f "data/gen$gen/book.epd" ]]; then
                kv "book" "data/gen$gen/book.epd (kept: datagen resumes from the games already played)"
            else
                python3 tools/gen_book.py --count "$(( (GAMES * (100 - RANDOM_OPENINGS) + 99) / 100 ))" --base "$BOOK" \
                    --seed "$((gen + 1))" --out "data/gen$gen/book.epd"
            fi
            bookopt=(--book "data/gen$gen/book.epd")
        fi
        ./tsumi datagen --net "$best" --threads "$THREADS" --games "$GAMES" --nodes "$NODES" "${bookopt[@]}" \
            --random-openings "$RANDOM_OPENINGS" --opening-min-score "$OPENING_MIN_SCORE" --opening-max-score "$OPENING_MAX_SCORE" \
            --random-move-pct "$RANDOM_MOVE_PCT" --random-move-minply "$RANDOM_MOVE_MINPLY" --random-move-maxply "$RANDOM_MOVE_MAXPLY" \
            --random-multi-pv "$RANDOM_MULTI_PV" --random-multi-pv-diff "$RANDOM_MULTI_PV_DIFF" --write-min-ply "$WRITE_MIN_PLY" \
            --seed "$((gen * 1000 + 1))" --out "data/gen$gen/data.bin" --log "data/gen$gen/games.log" --resume | tee -ai "logs/datagen_gen$gen.log"
        touch "data/gen$gen/done"
    else
        kv "data" "data/gen$gen/data.bin exists ($(( $(stat -c %s "data/gen$gen/data.bin") / 32 )) positions): not regenerated"
    fi
    data_time=$(( $(date +%s) - phase_start ))
    # How well the search scores predict the results (train/calibrate.py).
    rule "calibration"
    python3 train/calibrate.py "data/gen$gen/data.bin" --anchor-p "$ANCHOR_P" | tee -i "logs/calibrate_gen$gen.log"

    # Training on this generation's data and up to HISTORY - 1 earlier generations labelled by the same network.
    datafiles=("data/gen$gen/data.bin")
    this_source=$(cat "data/gen$gen/source")
    for ((prev = gen - 1; prev >= 0 && (HISTORY == 0 || ${#datafiles[@]} < HISTORY); prev--)); do
        [[ -f "data/gen$prev/data.bin" && -f "data/gen$prev/source" ]] || break
        [[ "$(cat "data/gen$prev/source")" == "$this_source" ]] || break
        datafiles+=("data/gen$prev/data.bin")
    done
    rule "training"
    phase_start=$(date +%s)
    if [[ -f "$trained" && -f "$net" ]]; then
        kv "candidate" "$net exists (its training is complete): not trained again"
    else
        gens=$(printf '%s\n' "${datafiles[@]}" | sed -E 's|^data/gen([0-9]+)/data\.bin$|\1|' | paste -sd, - | sed 's/,/, /g')
        kv "training on" "generation$([[ ${#datafiles[@]} -gt 1 ]] && echo s) $gens: the data labelled by the best network$([[ "$MAX_POSITIONS" -gt 0 ]] && echo ", at most $MAX_POSITIONS positions, newest first")"
        trainopt=(--lr "$LR_SCRATCH")
        if [[ "$RESUME" == 1 && -f "$best.pt" ]]; then trainopt=(--resume "$best.pt" --lr "$LR"); fi
        if [[ "$KEEP_BEST" == 1 ]]; then trainopt+=(--keep-best); fi
        if [[ "$RESULT_FILTER" != 1 ]]; then trainopt+=(--no-result-filter); fi
        python3 train/train.py --data "${datafiles[@]}" --out "$net" --hidden "$HIDDEN" --epochs "$EPOCHS" --lambda "$LAMBDA" \
            --end-lambda "$END_LAMBDA" --loss-asymmetry "$LOSS_ASYMMETRY" --anchor-p "$ANCHOR_P" --max-positions "$MAX_POSITIONS" \
            --progress-every "$PROGRESS_EVERY" ${DEVICE:+--device "$DEVICE"} "${trainopt[@]}" | tee -i "logs/train_gen$gen.log"
        python3 train/verify_net.py --net "$net" --data "data/gen$gen/data.bin" --count 100
        touch "$trained"
    fi
    train_time=$(( $(date +%s) - phase_start ))

    rule "match"
    phase_start=$(date +%s)
    touch "logs/match_gen$gen.running"
    limitopt=(--nodes "$MATCH_NODES")
    if [[ -n "$MATCH_TC" ]]; then limitopt=(--tc "$MATCH_TC"); fi
    python3 tools/match.py --e1 ./tsumi --e2 ./tsumi --opt1 "EvalFile=$net" --opt2 "EvalFile=$best" --name1 "gen$((gen+1))" --name2 "best" \
        --games "$MATCH_GAMES" "${limitopt[@]}" ${MATCH_BOOK:+--book "$MATCH_BOOK" --book-share $((100 - MATCH_RANDOM_SHARE))} --concurrency "$CONCURRENCY" \
        --pgn "logs/match_gen$gen.pgn" | tee -i "logs/match_gen$gen.log"
    match_time=$(( $(date +%s) - phase_start ))
    read -r elo err <<< "$(match_result "$gen")"
    [[ -n "$elo" ]] || { echo "logs/match_gen$gen.log has no result" >&2; exit 1; }

    echo
    rule "generation $gen: summary"
    gen_summary "$gen" "$data_time" "$train_time" "$match_time"
    if [[ "$(verdict "$elo")" == accepted ]]; then
        promote "$gen"
        kv "decision" "ACCEPTED: $net is the new $best"
    else
        kv "decision" "REJECTED: $best stays"
    fi
    echo "$(verdict "$elo") $elo $err" > "$decision"
    rm -f "logs/match_gen$gen.running"
    kv "time"     "$(elapsed "$gen_start") for this generation"
done
show_skipped
