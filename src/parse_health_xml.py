import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree as ET
from config import XML_FILE


def extract_records(xml_path=XML_FILE):
    """
    Lädt alle Record-Elemente aus der Apple Health XML.
    Gibt eine Liste von Dictionaries zurück.

    Jedes Dict enthält: type, value, unit, startDate, endDate.
    """
    records = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.XMLSyntaxError as e:
        print(f"Fehler beim Parsen der XML: {e}")
        return records
    except OSError as e:
        print(f"Datei nicht gefunden oder nicht lesbar: {e}")
        return records

    for record in root.iter("Record"):
        records.append({
            "type":      record.get("type"),
            "value":     record.get("value"),
            "unit":      record.get("unit"),
            "startDate": record.get("startDate"),
            "endDate":   record.get("endDate"),
        })

    print(f"{len(records)} Records geladen.")
    return records


def filter_records(records, record_type):
    """
    Filtert Records nach einem bestimmten HK-Typ.

    Beispiel-Typen:
        HKQuantityTypeIdentifierRestingHeartRate
        HKQuantityTypeIdentifierHeartRateVariabilitySDNN
        HKCategoryTypeIdentifierSleepAnalysis
        HKQuantityTypeIdentifierAppleExerciseTime
    """
    filtered = [r for r in records if r.get("type") == record_type]
    print(f"{len(filtered)} Records vom Typ '{record_type}' gefunden.")
    return filtered


if __name__ == "__main__":
    records = extract_records()
    aet_records = filter_records(records, "HKQuantityTypeIdentifierAppleExerciseTime")
    sa_records  = filter_records(records, "HKCategoryTypeIdentifierSleepAnalysis")
    rhr_records = filter_records(records, "HKQuantityTypeIdentifierRestingHeartRate")
    hrv_records = filter_records(records, "HKQuantityTypeIdentifierHeartRateVariabilitySDNN")
    print(records[:3])
