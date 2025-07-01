#!/bin/bash

# Parse arguments
time="00:01:00"
conda="ml4mikc"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--time) time="$2"; shift ;;
        -c|--conda) conda="$2"; shift ;;
        -f|--file) file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate time format
if ! [[ "$time" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Invalid time format. Use HH:MM:SS."
    exit 1
fi

# Create temporary SBATCH script
SBATCH_SCRIPT=$(mktemp)
cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --time=$time
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --output=job_output_%j.txt

source "$(conda info --base)/etc/profile.d/conda.sh" 
conda activate $conda

python -u "$file"
EOF

# Submit the job
sbatch "$SBATCH_SCRIPT"

# Remove temporary SBATCH script
rm "$SBATCH_SCRIPT"