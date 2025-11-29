import torch
from datetime import datetime

# ============================================================================
# Utility Functions
# ============================================================================
def get_available_gpus(gpu_ids=None):
    """Get available GPU IDs"""
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(',')]
    return list(range(torch.cuda.device_count()))


def print_time():
    """Print current timestamp"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
