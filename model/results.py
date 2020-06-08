import numpy as np
import xml.etree.ElementTree as ET
import os

def read_results(loc):
	filename = os.path.join(loc, 'composite.xml')

	if not os.path.exists(filename):
		raise ValueError(f'File composite.xml not found in: {loc}')

	r = parse_xml(filename)

	return r

def parse_xml(xml_file):
	tree = ET.parse(xml_file)

	root = tree.getroot()

	c_results = [c for c in root if c.tag=='Result'] #c for children

	results = []
	for c in c_results: #loop over results tags
		r = {}
		for elem in c: #loop over results' children
			if elem.tag == 'Data':
				data = {}
				for e in elem:
					for entry in e:
						if entry.tag=='Value':
							data['value'] = float(entry.text)
						if entry.tag=='RawString':
							data['raw_values'] = [float(i) for i in entry.text.split(":")]
			else:
				data = elem.text

			r[elem.tag] = data
		results.append(r)

	return results