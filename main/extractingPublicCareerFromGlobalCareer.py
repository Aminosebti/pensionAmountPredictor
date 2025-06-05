import xml.etree.ElementTree as ET
import glob
import os

def strip_namespace(tag):
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def find_tag_by_localname(root, localname):
    for elem in root.iter():
        if strip_namespace(elem.tag) == localname:
            return elem
    return None

def extract_and_save_combined(xml_path, output_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        person_ident = find_tag_by_localname(root, "personIdentification")
        detailed_career = find_tag_by_localname(root, "publicDetailedCareer")
        nominal_amount = find_tag_by_localname(root, "nominalAmount")

        base_filename = os.path.splitext(os.path.basename(xml_path))[0]

        if person_ident is None or detailed_career is None:
            print(f"⚠️  {base_filename}: Skipped because required tags are not present")
            return

        # Remove extractTimestamp attribute if it exists
        detailed_career.attrib.pop("extractTimestamp", None)

        # Build new XML structure
        new_root = ET.Element("extractedData")
        new_root.append(person_ident)
        new_root.append(detailed_career)

        if nominal_amount is not None:
            new_root.append(nominal_amount)

        # Save output
        output_str = ET.tostring(new_root, encoding='unicode')
        out_path = os.path.join(output_dir, f"{base_filename}_extracted.xml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_str)

        print(f"✅ Saved combined XML: {out_path}")

    except Exception as e:
        print(f"❌ Error with {xml_path}: {e}")

def main():
    folder_path = "/home/aminesebti/Jiras/ai/globalCareerCollecting/run_5-6-2025/dataWithNominalAmount"
    output_dir = "/home/aminesebti/Jiras/ai/globalCareerCollecting/run_5-6-2025/extractedIdentificationPublicCareerAndNominalAmount"

    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))

    print(f"\n🔍 Found {len(xml_files)} XML files. Processing...\n")

    for xml_path in xml_files:
        extract_and_save_combined(xml_path, output_dir)

if __name__ == "__main__":
    main()
