import zipfile, xml.etree.ElementTree as ET
def read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml_content = z.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    return ' '.join([node.text for node in tree.findall('.//w:t', ns) if node.text])

print(read_docx('c:/Users/laksh/Desktop/IITD/ML/Project/Project Document.docx')[2800:6000])
