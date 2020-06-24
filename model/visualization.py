import matplotlib.pylab as plt
import glob
import os
import config, model

plt.ion()

def boxplots_binary(param_name, test_name):
	results = model.FullScan.read_full_scan_results(param_name)

	data = []
	for res in results:
		params = res[0]
		for test in res[1]:
			data.append()
			plt.boxplot(test['Data']['raw_values'],
						positions=[counter])