import tempfile
import numbers
from pathlib import Path
from transformers.integrations.integration_utils import logger

def new_wandb_on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
    if self._wandb is None:
        return
    if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
        from transformers import Trainer
        fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
        fake_trainer.is_deepspeed_enabled = False
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_trainer.save_model(temp_dir)
            metadata = (
                {
                    k: v
                    for k, v in dict(self._wandb.summary).items()
                    if isinstance(v, numbers.Number) and not k.startswith("_")
                }
                if not args.load_best_model_at_end
                else {
                    f"eval/{args.metric_for_best_model}": state.best_metric,
                    "train/total_floss": state.total_flos,
                }
            )
            logger.info("Logging model artifacts. ...")
            model_name = (
                f"model-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"model-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
            for f in Path(temp_dir).glob("*"):
                if f.is_file():
                    with artifact.new_file(f.name, mode="wb") as fa:
                        fa.write(f.read_bytes())
            self._wandb.run.log_artifact(artifact)
            

    
