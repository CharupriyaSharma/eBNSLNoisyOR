#!/bin/bash
#SBATCH -A def-vanbeek
#SBATCH -t 23:59:00
#SBATCH --mem=8G
#SBATCH --output=out.txt

# ---------------------------------------------------------------------
echo $1 $2 $3 $4 
echo "Starting run at: `date`"
# ---------------------------------------------------------------------

START=$(date +%s%N)
python ../score1.py $1 $2 $3 $4
END=$(date +%s%N)
DIFF1=$(( $END - $START ))
echo "GOBNILP with BIC took $DIFF1 n-seconds"

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
