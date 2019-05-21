import os
import json
import urllib
import urllib.request
import numpy as np

#from utils.config import get

def main():
	'''
	test_data = input("Is these for testing: ")

	if test_data:
		im_dir = os.path.join(get("DATA.DATA_PATH"), 'test_data', 'Images')
	else:
		im_dir = os.path.join(get("DATA.DATA_PATH"), 'Images')'''

	with open('eyesnap.json', encoding="utf8") as json_file:
		data = json.load(json_file)

	gazes = {'FORWARD_GAZE', 'LEFTWARD_GAZE', 'RIGHTWARD_GAZE', 'UPWARD_GAZE'}
	eyes = {'LEFT', 'RIGHT'}


	for case in data['results']:
		if 'LEFT_EYE_DIAGNOSIS' not in case or 'RIGHT_EYE_DIAGNOSIS' not in case:
			continue

		'''if case['updatedAt'] < date:
			continue'''

		left_dia = case['LEFT_EYE_DIAGNOSIS']
		right_dia = case['RIGHT_EYE_DIAGNOSIS']

		if left_dia == 'Normal' and right_dia == 'Normal':
			for gaze in gazes:
				try:
					if gaze in case and not np.any([case[gaze + '_' + eye + '_EYE_DIAGNOSIS_VISIBLE'] for eye in eyes]):
						im_name = 'data/healthy/images/' + case[gaze]['name'] + '.jpeg'
						if not os.path.isfile(im_name):
							url = case[gaze]['url']
							req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
							con = urllib.request.urlopen(req)
							with open(im_name, 'wb+') as f:
								f.write(con.read())
				except:
					continue

		if left_dia == 'Retinoblastoma':
			for gaze in gazes:
				try:
					if gaze in case and case[gaze + '_LEFT_EYE_DIAGNOSIS_VISIBLE']:
						im_name = 'data/unhealthy/left/' + case[gaze]['name'] + '.jpeg'
						if not os.path.isfile(im_name):
							url = case[gaze]['url']
							req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
							con = urllib.request.urlopen(req)
							with open(im_name, 'wb+') as f:
								f.write(con.read())
				except:
					continue

		if right_dia == 'Retinoblastoma':
			for gaze in gazes:
				try:
					if gaze in case and case[gaze + '_RIGHT_EYE_DIAGNOSIS_VISIBLE']:
						im_name = 'data/unhealthy/right/' + case[gaze]['name'] + '.jpeg'
						if not os.path.isfile(im_name):
							url = case[gaze]['url']
							req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
							con = urllib.request.urlopen(req)
							with open(im_name, 'wb+') as f:
								f.write(con.read())
				except:
					continue

if __name__ == "__main__":
	main()