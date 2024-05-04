STEPS=(400 800 1600 6000)
GCS_BUCKET="gs://tacc-backup"
BASE_CKPT_DIR="experiment_ckpts/tinyllama_expanded-2024-03-03-10-26-24"
DEST_DIR="experiment_ckpts/tinyllama_expanded-2024-03-03-10-26-24"
for STEP in "${STEPS[@]}"; do
  SOURCE_CKPT="$GCS_BUCKET/$BASE_CKPT_DIR/checkpoint-$STEP"
  DEST_CKPT="$DEST_DIR/"  # checkpoint-$STEP will be created automatically
  if [ ! -d "$DEST_CKPT" ]; then
    echo "Directory $DEST_CKPT does not exist, creating now."
    mkdir -p $DEST_CKPT
  fi
  echo "gcloud storage cp -r $SOURCE_CKPT $DEST_CKPT"  # check and execute this command manually
#   gcloud storage cp -r $SOURCE_CKPT $DEST_CKPT
done