# Train CLIPScoreFusion model on MBEIR dataset

# Initialize Conda
source /home/miniconda3/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

# Path to the codebase and config file
SRC="$HOME/mbeir/src"  # Absolute path to codebse /mbeir/src # <--- Change this to the path of your mbeir/src

# Path to models_script dir
SCRIPT_DIR="$SRC/models_scripts"

# Path to MBEIR data and MBEIR directory where we store the checkpoints, embeddings, etc.
MBEIR_DIR="/data/mbeir/" # <--- Change this to the MBEIR directory
MBEIR_DATA_DIR="/data/mbeir/mbeir_data/" # <--- Change this to the MBEIR data directory

# Path to config dir
MODEL="clip_union/clip_scorefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models_scripts/$MODEL"
SIZE="large"
MODE="train"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # <--- Change this to the CUDA devices you want to us
NPROC=8
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update config
CONFIG_PATH="$CONFIG_DIR/inbatch.yaml"
cd $SCRIPT_DIR
python exp_utils.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct False

# Change to model directory
cd $MODEL_DIR
SCRIPT_NAME="train.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Activate conda environment
conda activate clip_sf # <--- Change this to the name of your conda environment

# Run training command
python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --mbeir_dir "$MBEIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"