from unidecode import unidecode
from pathlib import Path
import sys

p=sys.argv[1]

for file in Path(p).glob("*.*"):
    file_name = Path(unidecode(str(file).lower()))
    file.rename(file_name)