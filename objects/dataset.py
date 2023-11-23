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
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
		self.data.drop(columns=['ID'], inplace=True)
		new_data = pd.DataFrame(columns=["patientID", "imageID", "Presence"])

		for idx in range(len(self.data)):
			path = self.folder_path+"/"+self.data.iloc[idx]["patientID"]+"/"+self.data.iloc[idx]["imageID"]+".png"
			if os.path.isfile(path):
				new_data.loc[len(new_data)] = self.data.iloc[idx]
		
		self.data = new_data
		
	
	def get_labeled(self, label):
		self.data = self.data[self.data['Presence'] == label]

	def __getitem__(self, idx):
		path = self.folder_path+"/"+self.data.iloc[idx]["patientID"]+"/"+self.data.iloc[idx]["imageID"]+".png"
		
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self.transform(image)
		return image


class CroppedDataset(QuironDataset):
	def __init__(self, folder_path, csv_name, transform, anti_folder, anti_csv):
		super().__init__(folder_path, csv_name, transform)
		anti_df = pd.read_csv(anti_folder + "/" + anti_csv)
		self.anti_patients = anti_df["ID"].apply(lambda x: x.split(".")[0] if "." in x else None).unique().tolist()
		self.process_csv()
		self.used_patients = []
	
	def process_csv(self):
		self.data = pd.DataFrame(columns=["patientID", "imageID", "Presence"])
		folders = self.raw_data[self.raw_data["DENSITAT"] == "NEGATIVA"]["CODI"].tolist()
		for folder in folders:
			if folder in self.anti_patients:
				continue
			if len(self.data) > 10000:
				break
			try:
				self.used_patients.append(folder)
				images = os.listdir(self.folder_path + "/" + folder+ "_1")
				for img in images:
					row = [folder, img, -1]
					self.data.loc[len(self.data)] = row
			except:
				print(f"PATH NOT FOUND {self.folder_path + '/' + folder+ '_1'}")

	def get_used_patients(self):
		return self.used_patients

class PatientDataset(QuironDataset):
	def __init__(self, folder_path, csv_name, transform, used_patients):
		super().__init__(folder_path, csv_name, transform)
		self.used_patients = used_patients
		self.patients = {}
		self.process_csv()
	
	def process_csv(self):
		self.data = pd.DataFrame(columns=["patientID", "imageID", "Presence"])
		folders = self.raw_data[~self.raw_data['CODI'].isin(self.used_patients)]['CODI'].tolist()
		for folder in folders:
			if len(self.data) > 200_000:
				break
			try:
				images = os.listdir(self.folder_path + "/" + folder+ "_1")
				for img in images:
					row = [folder, img, self.raw_data[self.raw_data['CODI'] == folder]['DENSITAT']]
					self.data.loc[len(self.data)] = row
					self.patients[len(self.data)] = folder
			except:
				print(f"PATH NOT FOUND {self.folder_path + '/' + folder+ '_1'}")
	
	def get_patients(self):
		return self.patients
	
	def get_patients_results(self):
		results = {}
		for patient in set(self.patients.values()):
			status = self.data[self.data['patientID'] == patient]['Presence'].values[0].iloc[-1]
			results[patient] = 1 if status != 'NEGATIVA' else 0
		return results

