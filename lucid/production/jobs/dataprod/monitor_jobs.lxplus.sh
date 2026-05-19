#!/bin/bash
# Monitor PhotonSim jobs on LXPLUS HTCondor.
#
# Counterpart of monitor_jobs.s3df.sh (which uses squeue). squeue and
# condor_q output are too different for one parser, so each cluster has
# its own monitor; the rest of the production pipeline is cluster-
# agnostic (see ../../../../docs/CLUSTER_ABSTRACTION.md).

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SHOW_ALL=false
WATCH_MODE=false
OUTPUT_DIR=""

while getopts "awo:h" opt; do
    case $opt in
        a) SHOW_ALL=true;;
        w) WATCH_MODE=true;;
        o) OUTPUT_DIR="$OPTARG";;
        h) echo "Usage: $0 [-a] [-w] [-o output_dir]"
           echo "  -a: Show all jobs (default: only PhotonSim jobs)"
           echo "  -w: Watch mode — refresh every 30 seconds"
           echo "  -o: Check specific output directory for results"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done

show_jobs() {
    clear
    echo -e "${GREEN}=== PhotonSim Job Monitor (HTCondor) ===${NC}"
    echo -e "Time: $(date)"
    echo ""

    echo -e "${YELLOW}Running Jobs:${NC}"
    if [ "$SHOW_ALL" = true ]; then
        condor_q $USER -nobatch
    else
        condor_q $USER -nobatch | grep -E "(ID|photonsim|train_|siren_|smax_)" || echo "(none)"
    fi

    echo ""
    echo -e "${YELLOW}Job Statistics:${NC}"
    RUNNING=$(condor_q $USER -hold:false -running -af ClusterId 2>/dev/null | wc -l)
    IDLE=$(condor_q $USER -idle -af ClusterId 2>/dev/null | wc -l)
    HELD=$(condor_q $USER -hold -af ClusterId 2>/dev/null | wc -l)
    echo -e "Jobs running: ${GREEN}${RUNNING:-0}${NC}"
    echo -e "Jobs idle:    ${YELLOW}${IDLE:-0}${NC}"
    echo -e "Jobs held:    ${RED}${HELD:-0}${NC}"
    echo -e "Total:        ${BLUE}$((RUNNING + IDLE + HELD))${NC}"

    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo -e "${YELLOW}Output Files in $OUTPUT_DIR:${NC}"
        for energy_dir in $(find "$OUTPUT_DIR" -type d -name "*MeV" 2>/dev/null | sort -V); do
            ENERGY=$(basename "$energy_dir")
            ROOT_COUNT=$(find "$energy_dir" -name "*.root" 2>/dev/null | wc -l)
            LOG_COUNT=$(find "$energy_dir" -name "job-*.out" 2>/dev/null | wc -l)
            if [ "$ROOT_COUNT" -gt 0 ] || [ "$LOG_COUNT" -gt 0 ]; then
                echo "  $ENERGY: $ROOT_COUNT ROOT files, $LOG_COUNT job logs"
            fi
        done

        echo ""
        echo -e "${YELLOW}Recently completed (last 5):${NC}"
        find "$OUTPUT_DIR" -name "job-*.out" -type f -mmin -60 2>/dev/null | \
            xargs -I {} sh -c 'echo -n "  "; basename {} | cut -d. -f1; grep "Job ended" {} | tail -1' | \
            tail -5
    fi
}

if [ "$WATCH_MODE" = true ]; then
    echo "Entering watch mode. Press Ctrl+C to exit."
    while true; do
        show_jobs
        sleep 30
    done
else
    show_jobs
fi
