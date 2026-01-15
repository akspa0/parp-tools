from pathlib import Path
import os

ROOT = r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar"
DS = Path(ROOT) / "dataset"

print(f"Root: {ROOT}")
print(f"Dataset Path: {DS}")
print(f"Exists: {DS.exists()}")
print(f"Is Dir: {DS.is_dir()}")

print("Listing dir via os.listdir:")
try:
    print(os.listdir(str(DS))[:5])
except Exception as e:
    print(e)

print("Globbing *.json:")
try:
    files = list(DS.glob("*.json"))
    print(f"Found {len(files)} files.")
    if files:
        print(f"Example: {files[0]}")
except Exception as e:
    print(e)
