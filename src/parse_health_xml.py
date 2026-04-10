import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree as ET
from config import XML_FILE


def extract_records(xml_path=XML_FILE) -> list:
    """
    Parses all Record elements from an Apple Health export XML.

    Returns a list of dicts with keys: type, value, unit, startDate, endDate.
    Returns an empty list if the file is missing or malformed.
    """
    records = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.XMLSyntaxError as e:
        print(f"XML parse error: {e}")
        return records
    except OSError as e:
        print(f"Could not open file: {e}")
        return records

    for record in root.iter("Record"):
        records.append({
            "type":      record.get("type"),
            "value":     record.get("value"),
            "unit":      record.get("unit"),
            "startDate": record.get("startDate"),
            "endDate":   record.get("endDate"),
        })

    print(f"{len(records)} records loaded.")
    return records


def filter_records(records: list, record_type: str) -> list:
    """
    Filters a list of record dicts by Apple Health type identifier.

    Common types:
        HKQuantityTypeIdentifierRestingHeartRate
        HKQuantityTypeIdentifierHeartRateVariabilitySDNN
        HKCategoryTypeIdentifierSleepAnalysis
        HKQuantityTypeIdentifierAppleExerciseTime
    """
    filtered = [r for r in records if r.get("type") == record_type]
    print(f"{len(filtered)} records of type '{record_type}'.")
    return filtered


if __name__ == "__main__":
    records = extract_records()
    filter_records(records, "HKQuantityTypeIdentifierAppleExerciseTime")
    filter_records(records, "HKCategoryTypeIdentifierSleepAnalysis")
    filter_records(records, "HKQuantityTypeIdentifierRestingHeartRate")
    filter_records(records, "HKQuantityTypeIdentifierHeartRateVariabilitySDNN")
    print(records[:3])
