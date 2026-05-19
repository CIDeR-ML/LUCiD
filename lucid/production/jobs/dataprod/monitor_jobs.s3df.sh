#!/bin/bash
# Script to monitor PhotonSim jobs on S3DF

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
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
           echo "  -w: Watch mode - refresh every 30 seconds"
           echo "  -o: Check specific output directory for results"
           exit 0;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    esac
done

function show_jobs() {
    clear
    echo -e "${GREEN}=== PhotonSim Job Monitor ===${NC}"
    echo -e "Time: $(date)"
    echo ""
    
    # Show running jobs
    echo -e "${YELLOW}Running Jobs:${NC}"
    if [ "$SHOW_ALL" = true ]; then
        squeue -u $USER
    else
        squeue -u $USER | grep -E "(JOBID|photonsi|multiparticle)"
    fi

    echo ""

    # Show job statistics
    echo -e "${YELLOW}Job Statistics:${NC}"
    RUNNING=$(squeue -u $USER -h -t RUNNING | grep -E "photonsi|multiparticle" | wc -l || true)
    PENDING=$(squeue -u $USER -h -t PENDING | grep -E "photonsi|multiparticle" | wc -l || true)
    RUNNING=${RUNNING:-0}
    PENDING=${PENDING:-0}
    TOTAL=$((RUNNING + PENDING))

    echo -e "PhotonSim jobs running: ${GREEN}$RUNNING${NC}"
    echo -e "PhotonSim jobs pending: ${YELLOW}$PENDING${NC}"
    echo -e "Total PhotonSim jobs: ${BLUE}$TOTAL${NC}"
    
    # Check output directory if specified
    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo -e "${YELLOW}Output Files in $OUTPUT_DIR:${NC}"
        
        # Count ROOT files by energy
        for energy_dir in $(find "$OUTPUT_DIR" -type d -name "*MeV" 2>/dev/null | sort -V); do
            ENERGY=$(basename "$energy_dir")
            ROOT_COUNT=$(find "$energy_dir" -name "*.root" 2>/dev/null | wc -l)
            LOG_COUNT=$(find "$energy_dir" -name "job-*.out" 2>/dev/null | wc -l)
            
            if [ $ROOT_COUNT -gt 0 ] || [ $LOG_COUNT -gt 0 ]; then
                echo "  $ENERGY: $ROOT_COUNT ROOT files, $LOG_COUNT job logs"
            fi
        done
        
        # Show recent completions
        echo ""
        echo -e "${YELLOW}Recently completed (last 5):${NC}"
        find "$OUTPUT_DIR" -name "job-*.out" -type f -mmin -60 2>/dev/null | \
            xargs -I {} sh -c 'echo -n "  "; basename {} | cut -d. -f1; grep "Job ended at:" {} | tail -1' | \
            tail -5
    fi
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
    echo "Entering watch mode. Press Ctrl+C to exit."
    while true; do
        show_jobs
        sleep 30
    done
else
    show_jobs
fi