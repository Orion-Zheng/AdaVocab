DIST_CKPT="experiment_ckpts/tinyllama_expanded_frez_embed-2024-04-16-010251/checkpoint-3"
python dist_env_tools/ds_to_universal.py --input_folder ${DIST_CKPT} \
                                         --output_folder ${DIST_CKPT}_universal \
                                         --num_extract_workers 10 \
                                         --num_merge_workers 10 \