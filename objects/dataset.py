import cv2
import os
import pandas as pd
from torch.utils.data import Dataset


class QuironDataset(Dataset):
	def __init__(self, folder_path, csv_name, transform):
		self.folder_path = folder_path
		self.raw_data = pd.read_csv(folder_path+"/"+csv_name)
		self.transform = transform

	def __getitem__(self, idx):
		image = cv2.imread(self.folder_path+"/"+self.data.iloc[idx]["patientID"]+"_1/"+self.data.iloc[idx]["imageID"])
		image = self.transform(image)

		return image

	def __len__(self):
		return len(self.data)

class AnnotatedDataset(QuironDataset):
	def __init__(self, folder_path, csv_name, transform, label):
		super().__init__(folder_path, csv_name, transform)
		if label not in [-1, 1]:
			raise ValueError(f'Label for AnnotatedDataset must be -1 or 1 and not {label}')
		self.data = self.raw_data
		self.get_labeled(label)
		self.process_csv()

	def process_csv(self):
		self.data["patientID"] = self.data["ID"].apply(lambda x: x.split(".")[0] if "." in x else None)
		self.data["imageID"] = self.data["ID"].apply(lambda x: x.split(".")[1] if "." in x else None)
	
	def get_labeled(self, label):
		self.data = self.data[self.data['Presence'] == label]

	def __getitem__(self, idx):
		image = cv2.imread(self.folder_path+"/"+self.data.iloc[idx]["patientID"]+"/"+self.data.iloc[idx]["imageID"]+".png")
		image = self.transform(image)
		return image


class CroppedDataset(QuironDataset):
	def __init__(self, folder_path, csv_name, transform):
		super().__init__(folder_path, csv_name, transform)
		self.process_csv()
	
	def process_csv(self):
		self.data = pd.DataFrame(columns=["patientID", "imageID", "Presence"])
		folders = self.raw_data[self.raw_data["DENSITAT"] == "NEGATIVA"]["CODI"].tolist()
		for folder in folders:
			if len(self.data)>10000:
				break
			try:
				images = os.listdir(self.folder_path + "/" + folder+ "_1")
				for img in images:
					row = [folder, img, -1]
					self.data.loc[len(self.data)] = row
			except:
				print(f"PATH NOT FOUND {self.folder_path + '/' + folder+ '_1'}")





