import os
import logging

__all__ = ["Settings"]

# Simple class to store default settings for execution
class Settings():
    save = True
    
    include_errors = True

    seed = 42
    num_threads = os.cpu_count()
    num_steps = 25000
