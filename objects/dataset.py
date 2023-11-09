import io
import os
import pandas as pd
from torch.utils.data import Dataset


class QuironDataset(Dataset):
	def __init__(self, folder_path, csv_name):
		self.folder_path = folder_path
		self.raw_data = pd.read_csv(folder_path+"/"+csv_name)

	def __getitem__(self, idx):
		return self.data.iloc[idx], io.imread(self.folder_path+"/"+self.data["patientId"]+"/"+self.data["imageID"])
	def __len__(self):
		return len(self.df)
	

class AnnotatedDataset(QuironDataset):
	def __init__(self, folder_path, csv_name):
		super().__init__(folder_path, csv_name)
		self.data = self.raw_data
		self.process_csv()

	def process_csv(self):
		self.data["patientId"] = self.data["ID"].split(".")[0]
		self.data["imageID"] = self.data["ID"].split(".")[1]


class CroppedDataset(QuironDataset):
	def __init__(self, folder_path, csv_name):
		super().__init__(folder_path, csv_name)
		self.process_csv()
	
	def process_csv(self):
		self.data = pd.Dataframe(columns=["parientID", "imageID", "Presence"])
		folders = self.raw_data[self.raw_data["DENSITAT"] == "NEGATIVA"]["CODI"].tolist()
		for folder in folders:
			images = os.listdir(self.folder_path + "/" + folder)
			row = [folder, images, -1]
			self.data.iloc[len(self.data)] = row




