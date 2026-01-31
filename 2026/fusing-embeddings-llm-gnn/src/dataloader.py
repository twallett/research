import os
from pathlib import Path

data_dir = Path.cwd().parent / "preprocessed_data"
os.chdir(data_dir)
link = "wget -O 'data_v1.zip' 'https://gwu.box.com/shared/static/6td89qmbuikobs8x7ajii3o6adfzsyhm.zip'"

os.system(f"{link}")
os.system("unzip data_v1.zip")