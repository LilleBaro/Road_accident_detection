import sys, subprocess
import torch
try:
    import ultralytics
except Exception as e:
    ultralytics = None
print("Python:", sys.version.splitlines()[0])
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
try:
    print("torch.version.cuda:", torch.version.cuda)
except:
    pass
print("ultralytics:", getattr(ultralytics, "__version__", "not installed"))
