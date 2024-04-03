import torch

def get_device():
    try:
        if torch.cuda.is_available():
            return '0'
        elif torch.backends.mps.is_available():
            return 'mps'
    except:
        pass
    
    return 'cpu'