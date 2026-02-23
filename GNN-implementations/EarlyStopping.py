from enum import Enum


class EarlyStopping:
    
    def __init__(self, min_delta:float, patience:int):
        self.min_delta = min_delta
        self.patience = patience
        self.low = None
        self.counter = 0
        
    def __call__(self, val_loss):
        
        if self.low is None:
            self.low = val_loss
            return False
        
        if val_loss < self.low - self.min_delta:
            self.counter = 0
            self.low = val_loss
            return False
        
        if val_loss < self.low:
            
            self.counter += 1
            if self.counter >= self.patience:
                return True
            self.low = val_loss
            return False
    
    
        self.counter += 1
        return self.counter >= self.patience
