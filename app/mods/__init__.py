""" mods module"""

import sys
from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent.parent.resolve())
app_path = str(Path(__file__).absolute().parent.parent.resolve())
sys.path.append(root_path)
sys.path.append(app_path) # add root and app project path to environment -> permits module import 