import xml.etree.ElementTree as ET

def extract_records(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    records = []
    for record in root.findall('.//record'):
        record_data = {}
        for field in record:
            record_data[field.tag] = field.text
        records.append(record_data)
    
    return records