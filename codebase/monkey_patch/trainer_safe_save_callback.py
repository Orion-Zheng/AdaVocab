import os
import subprocess
from datetime import datetime, timedelta

from transformers import TrainerCallback

from codebase.dist_logging import get_dist_logger
logger = get_dist_logger()

DEFAULT_SAFE_MINUTES = 5

# Function to execute qstat and return its output
def get_qstat_output(job_id):
    cmd = ['qstat', '-f', job_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# Function to parse qstat output for start time and wall time
def parse_times(qstat_output):
    lines = qstat_output.split('\n')
    times = {}
    for line in lines:
        if 'Resource_List.walltime' in line:
            times['walltime'] = line.split('=')[1].strip()
        elif 'stime' in line:
            times['stime'] = line.split('=')[1].strip()
    return times

def get_expect_end_time():
    # Assuming job_id is available
    job_id = os.getenv('PBS_JOBID')
    qstat_output = get_qstat_output(job_id)
    times = parse_times(qstat_output)

    # Calculate end time
    start_time = datetime.strptime(times['stime'], '%a %b %d %H:%M:%S %Y')
    hours, minutes, seconds = map(int, times['walltime'].split(':'))
    walltime = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    end_time = start_time + walltime
    return end_time

def test():
    # Assuming the safe time is 80 minutes: 
    # After each step, if trainer find that there are less than 80 minutes left, the model should save the checkpoint.
    safe_time = 80
    end_time = get_expect_end_time()
    adjusted_end_time = end_time - timedelta(minutes=safe_time)
    current_time = datetime.now()
    print(current_time)
    print(f"Adjusted end time ({safe_time} minutes earlier): {adjusted_end_time}")
    if current_time > adjusted_end_time:
        print("Alert: The current time has surpassed the adjusted end time.")
    else:
        print("The current time has not yet surpassed the adjusted end time.")

class SafeSavingCallback_NSCC(TrainerCallback):
    safe_minutes = DEFAULT_SAFE_MINUTES
    def __init__(self):
        self.end_time = get_expect_end_time()
        self.safe_must_save = self.end_time - timedelta(minutes=self.safe_minutes)
        self.already_save = False
        
    def on_step_end(self, args, state, control, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take several inputs.
        args: TrainingArguments, state: TrainerState, control: TrainerControl
        """
        current_time = datetime.now()
        if current_time > self.safe_must_save and not self.already_save:
            control.should_save = True
            logger.info('Reach Safe Saving Time, set `control.should_save = True`')
            self.already_save = True
            return control
        
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control
    
if __name__ == '__main__':
    test()