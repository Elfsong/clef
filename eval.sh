# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Data: 2022-03-25

export PROJECT=lucky-re
export ZONE=europe-west4-a
export TPU_NAME=instance-1
export BUCKET=gs://nlp_base/mingzhe/mt5

declare -a sizes=("xl")

TASK="clef_all"
EVAL_STEPS=1002000

for SIZE in "${sizes[@]}"; do
    PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
    MODEL_DIR="${BUCKET}/${TASK}/${SIZE}"
    
    # Run eval
    python -m t5.models.mesh_transformer_main \
        --module_import="clef_tasks" \
        --tpu="${TPU_NAME}" \
        --gcp_project="${PROJECT}" \
        --tpu_zone="${ZONE}" \
        --model_dir="${MODEL_DIR}" \
        --gin_file="dataset.gin" \
        --gin_file="${MODEL_DIR}/operative_config.gin" \
        --gin_file="eval.gin" \
        --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
        --gin_param="MIXTURE_NAME = '${TASK}'" \
        --gin_param="utils.run.dataset_split = 'dev'" \
        --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
        --gin_param="utils.run.eval_checkpoint_step=${EVAL_STEPS}" \
        --t5_tfds_data_dir="${BUCKET}/t5-tfds"
done
