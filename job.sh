#!/bin/bash

# Parse arguments
unit="gpu_a100"
time="00:01:00"
conda="ml4mikc"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--unit) unit="$2"; shift ;;
        -t|--time) time="$2"; shift ;;
        -c|--conda) conda="$2"; shift ;;
        -f|--file) file="$2"; shift ;;
        --*)
            if [[ -n "$2" && "$2" != --* ]]; then
                sbatch_args+=("$1=$2")
                shift
            else
                sbatch_args+=("$1")
            fi
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate arguments
if ! [[ "$unit" =~ ^(gpu_a100|gpu_h100|genoa|rome)$ ]]; then
    echo "Invalid processing unit unit specified. Check accinfo for valid options."
    exit 1
elif [[ "$unit" =~ ^(gpu_a100|gpu_h100)$ && ! "${sbatch_args[*]}" =~ --gpus= ]]; then
    sbatch_args+=("--gpus=1")
fi

if ! [[ "$time" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Invalid time format. Use HH:MM:SS."
    exit 1
fi

if [[ -z "$file" ]]; then
    echo "File to run is required. Use -f or --file to specify the file."
    exit 1
fi  

if ! conda env list | grep -q "$conda"; then
    echo "Conda environment '$conda' does not exist."
    exit 1
fi

# Parameters for SBATCH
echo "Processing unit set to: $unit"
echo "Time set to: $time"
echo "Conda environment set to: $conda"
echo "File to run: $file"
echo "Extra SBATCH arguments: ${sbatch_args[*]}"

# Add SBATCH script header
SBATCH_SCRIPT=$(mktemp)
cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash

#SBATCH --partition=$unit
#SBATCH --nodes=1
#SBATCH --time=$time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
EOF

# Add extra SBATCH arguments
for arg in "${sbatch_args[@]}"; do
    echo "#SBATCH $arg" >> "$SBATCH_SCRIPT"
done

# Add rest of the script
cat <<EOF >> "$SBATCH_SCRIPT"
echo "Job started at: \$(date '+%Y-%m-%d %H:%M:%S')"

source /home/asanchez/chonky/tools/CCP4/ccp4-9/bin/ccp4.setup-sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $conda

python -u $file
EOF

# Submit the job
sbatch "$SBATCH_SCRIPT"
