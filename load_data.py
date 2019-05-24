import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import random



def augment_and_store(data, labels):
	new_data, new_labels = [], []
	for eye, label in zip(data, labels):
		# cv2.namedWindow('augment_and_store', cv2.WINDOW_NORMAL)
		# cv2.imshow('augment_and_store', eye)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		new_data.append(cv2.resize(eye[int(eye.shape[0]/10):int(-1 * eye.shape[0]/10), int(eye.shape[1]/10):int(-1 * eye.shape[1]/10)], (224,224)))
		# new_data.append(cv2.resize(eye[int(eye.shape[0]/10):int(-1 * eye.shape[0]/10), int(eye.shape[1]/10):int(-1 * eye.shape[1]/10)], (100,48)))
		# temp = cv2.resize(eye[int(eye.shape[0]/10):int(-1 * eye.shape[0]/10), int(eye.shape[1]/10):int(-1 * eye.shape[1]/10)], (224,224))
		
		# cv2.namedWindow('augment_and_store', cv2.WINDOW_NORMAL)
		# cv2.imshow('augment_and_store', temp)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		new_labels.append(label)
		new_data.append(eye)
		new_labels.append(label)
		new_data.append(np.flipud(eye))
		new_labels.append(label)
		new_data.append(np.fliplr(eye))
		new_labels.append(label)
		new_data.append(np.flipud(np.fliplr(eye)))
		new_labels.append(label)
		for x in range(1, 10, 2):
			gauss = np.random.normal(0, 1, eye.shape) * 10
			noise = cv2.add(new_data[-1 * x], gauss, dtype=cv2.CV_8UC3)
			noise[noise > 255] = 255
			noise[noise < 0] = 0
			new_data.append(noise)
			new_labels.append(label)
	return new_data, new_labels


def load_image(folder, image_path, label, data, labels):
	xml_path = folder + 'annotations/' + image_path.split('.')[0] + '.xml'
	if not os.path.exists(xml_path):
		return
	image = cv2.imread(folder + image_path)
	root = ET.parse(xml_path).getroot()
	for child in root:
		if child.tag == 'object':
			bbox = [int(child[-1][0].text), int(child[-1][1].text), int(child[-1][2].text), int(child[-1][3].text)]
			data.append(cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (224, 224)))
			# data.append(cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (100, 48)))
			labels.append(label)


def normalize_images(data):
	mean = np.mean(data, axis=(0, 1, 2))
	std = np.std(data, axis=(0, 1, 2))
	return (data - mean) / std

def load_data():

	#load data
	healthy_data, unhealthy_data = [], []
	healthy_labels, unhealthy_labels = [], []
	print('Loading Cancer Images')
	for image_path in os.listdir('data/unhealthy/left/'):
		if '.jpeg' not in image_path:
			continue
		load_image('data/unhealthy/left/', image_path, 1, unhealthy_data, unhealthy_labels)
	for image_path in os.listdir('data/unhealthy/right/'):
		if '.jpeg' not in image_path:
			continue
		load_image('data/unhealthy/right/', image_path, 1, unhealthy_data, unhealthy_labels)
	print('Loading Normal Images')
	for image_path in os.listdir('data/healthy/'):
		if '.jpeg' not in image_path:
			continue
		load_image('data/healthy/', image_path, 0, healthy_data, healthy_labels)

	#balance classes and split train/test
	healthy_data, healthy_labels = healthy_data[:len(unhealthy_data)], healthy_labels[:len(unhealthy_labels)]
	ind = int(len(unhealthy_data) / 10) * 8
	train_data, train_labels = healthy_data[:ind] + unhealthy_data[:ind], healthy_labels[:ind] + unhealthy_labels[:ind]
	test_data, test_labels = healthy_data[ind:] + unhealthy_data[ind:], healthy_labels[ind:] + unhealthy_labels[ind:]

	#shuffle data
	temp = list(zip(train_data, train_labels))
	# random.shuffle(temp)
	train_data, train_labels = zip(*temp)
	train_data, train_labels = np.array(train_data), np.array(train_labels)
	temp = list(zip(test_data, test_labels))
	# random.shuffle(temp)
	test_data, test_labels = zip(*temp)
	test_data, test_labels = np.array(test_data), np.array(test_labels)

	#augment and normalize train set
	train_data, train_labels = augment_and_store(train_data, train_labels)
	train_data = normalize_images(train_data)
	test_data = normalize_images(test_data)
	
	# for x in range(3):
	# 	cv2.imshow('d', train_data[x])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	# 	cv2.imshow('d', test_data[x])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	#load unknown data
	unknown_data = []
	unknown_labels = []
	raw_unknown_data = []
	raw_unknown_labels = []
	for image_path in os.listdir('data/test/unhealthy/'):
		if '.jpeg' not in image_path:
			continue
		load_image('data/test/unhealthy/', image_path, 1, unknown_data, unknown_labels)
		load_image('data/test/unhealthy/', image_path, 1, raw_unknown_data, raw_unknown_labels)
	for image_path in os.listdir('data/test/healthy/'):
		if '.jpeg' not in image_path:
			continue
		load_image('data/test/healthy/', image_path, 0, unknown_data, unknown_labels)
		load_image('data/test/healthy/', image_path, 0, raw_unknown_data, raw_unknown_labels)
	print('Data loaded')
	unknown_data = normalize_images(unknown_data)

	return train_data, train_labels, test_data, test_labels, unknown_data, unknown_labels, raw_unknown_data