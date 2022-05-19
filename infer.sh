# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Data: 2022-04-06

export PROJECT=lucky-re
export ZONE=europe-west4-a
export TPU_NAME=instance-1
export BUCKET=gs://nlp_base/mingzhe/mt5

PRETRAIN_TASK="clef_1A_1B_1D_1E_1F_train_dev"
SIZE="xl"
INFER_STEPS=1002800

MODEL_DIR="${BUCKET}/${PRETRAIN_TASK}/${SIZE}"

# Run infer
python -m t5.models.mesh_transformer_main \
    --module_import="clef_tasks" \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="sample_decode.gin" \
    --gin_param="input_filename = '/home/dumingzhex/clef/CLEF_data/1A_checkworthy/spanish/test.tsv'"\
    --gin_param="output_filename = './CLEF_infer_output/spanish'"\
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="infer_checkpoint_step=${INFER_STEPS}" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds"

  
