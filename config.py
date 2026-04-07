from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
XML_FILE = DATA_RAW / "Export.xml"