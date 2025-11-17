import os
import shutil
from pathlib import Path
import nltk

print("=== NLTK FULL AUTO-FIX SCRIPT STARTED ===")

# -------------------------------------------------------
# 1. Define project NLTK directory
# -------------------------------------------------------
project_nltk = Path(r"D:\Projects\smart_resume_analyzer\nltk_data")

tokenizers_path = project_nltk / "tokenizers"
corpora_path = project_nltk / "corpora"

punkt_path = tokenizers_path / "punkt"
punkt_tab_path = tokenizers_path / "punkt_tab"
wordnet_path = corpora_path / "wordnet"

# Create directory structure
for p in [project_nltk, tokenizers_path, corpora_path]:
    p.mkdir(parents=True, exist_ok=True)

print("[OK] Project NLTK directory verified.")

# -------------------------------------------------------
# 2. Download NLTK resources
# -------------------------------------------------------
print("\n[INFO] Downloading punkt, punkt_tab, wordnet...")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
print("[OK] Downloads complete.")

# -------------------------------------------------------
# 3. Locate global NLTK data
# -------------------------------------------------------
global_nltk_path = Path(nltk.data.find("tokenizers/punkt")).parent.parent
print(f"\n[INFO] Global NLTK path: {global_nltk_path}")

global_punkt = global_nltk_path / "tokenizers" / "punkt"
global_punkt_tab = global_nltk_path / "tokenizers" / "punkt_tab"
global_wordnet = global_nltk_path / "corpora" / "wordnet"

# -------------------------------------------------------
# 4. Recursive folder copy helper
# -------------------------------------------------------
def copy_folder(src: Path, dst: Path):
    if not src.exists():
        print(f"[WARN] Missing source folder: {src}")
        return

    # Make sure the destination exists
    dst.mkdir(parents=True, exist_ok=True)
    
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            print(f"[DIR] Copied directory: {item.name}")
        else:
            shutil.copy2(item, target)
            print(f"[FILE] Copied file: {item.name}")

    print(f"[OK] Completed copy: {src.name}")

# -------------------------------------------------------
# 5. Copy punkt, punkt_tab, wordnet
# -------------------------------------------------------
copy_folder(global_punkt, punkt_path)
copy_folder(global_punkt_tab, punkt_tab_path)
copy_folder(global_wordnet, wordnet_path)

# -------------------------------------------------------
# 6. Add your project NLTK dir to search path
# -------------------------------------------------------
nltk.data.path.append(str(project_nltk))
print("\n[OK] Added to NLTK search path:", nltk.data.path)

# -------------------------------------------------------
# 7. Test: tokenization + lemmatization
# -------------------------------------------------------
print("\n[TEST] Checking sample tokenization and lemmatization...")

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    print("Tokenize:", word_tokenize("This is a test."))
    wnl = WordNetLemmatizer()
    print("Lemmatize:", wnl.lemmatize("running"))
    print("[SUCCESS] NLTK is fully operational!")
except Exception as e:
    print("[ERROR] Test failed:", e)

print("\n=== NLTK FULL AUTO-FIX COMPLETE ===")
