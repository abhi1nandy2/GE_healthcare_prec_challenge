import time
import pandas as pd
# time.sleep(54000)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms,datasets, models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torchvision
import pickle
from sklearn import model_selection
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import copy

from sklearn.metrics import confusion_matrix

import sys
from zipfile import ZipFile
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

print(torch.cuda.get_device_name(0))

Tensor = torch.cuda.FloatTensor

# df = pd.read_csv('')
num_classes = 2
WL = int(sys.argv[1])
in_channels=1
BatchSize = 16

class SegNet_enc(nn.Module):
	def __init__(self, in_channels):
		super(SegNet_enc, self).__init__()
		batchNorm_momentum = 0.1

		self.conv11 = nn.Conv2d(in_channels, 8, kernel_size=(1,3), padding=(0,1))
		self.bn11 = nn.BatchNorm2d(8, momentum= batchNorm_momentum)
		self.conv12 = nn.Conv2d(8, 8, kernel_size=(1,3), padding=(0,1))
		self.bn12 = nn.BatchNorm2d(8, momentum= batchNorm_momentum)

		self.conv21 = nn.Conv2d(8, 16, kernel_size=(1,3), padding=(0,1))
		self.bn21 = nn.BatchNorm2d(16, momentum= batchNorm_momentum)
		self.conv22 = nn.Conv2d(16, 16, kernel_size=(1,3), padding=(0,1))
		self.bn22 = nn.BatchNorm2d(16, momentum= batchNorm_momentum)

		self.conv31 = nn.Conv2d(16, 32, kernel_size=(1,3), padding=(0,1))
		self.bn31 = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
		self.conv32 = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1))
		self.bn32 = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
		self.conv33 = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1))
		self.bn33 = nn.BatchNorm2d(32, momentum= batchNorm_momentum)

		self.conv41 = nn.Conv2d(256, 512, kernel_size=(1,3), padding=(0,1))
		self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv42 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv43 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv51 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv52 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv53 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv53d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv52d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv51d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv43d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv42d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv41d = nn.Conv2d(512, 256, kernel_size=(1,3), padding=(0,1))
		self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

		self.conv33d = nn.Conv2d(256, 256, kernel_size=(1,3), padding=(0,1))
		self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
		self.conv32d = nn.Conv2d(256, 256, kernel_size=(1,3), padding=(0,1))
		self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
		self.conv31d = nn.Conv2d(256,  128, kernel_size=(1,3), padding=(0,1))
		self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

		self.conv22d = nn.Conv2d(128, 128, kernel_size=(1,3), padding=(0,1))
		self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
		self.conv21d = nn.Conv2d(128, 64, kernel_size=(1,3), padding=(0,1))
		self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

		self.conv12d = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
		self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
		self.conv11d = nn.Conv2d(64, in_channels, kernel_size=(1,3), padding=(0,1))
		self.fc1 = nn.Linear(in_features = 32*int(WL/8), out_features = int(WL/16), bias = True)
		self.fc2 = nn.Linear(in_features = 32*int(WL/8), out_features = num_classes, bias = True)

	def forward(self, x):
		x11 = F.relu(self.bn11(self.conv11(x)))
		x12 = F.relu(self.bn12(self.conv12(x11)))
		# print(x12.shape)
		# exit()
		x1p, id1 = F.max_pool2d(x12,kernel_size=(1,2), stride=(1,2),return_indices=True)

		# Stage 2
		x21 = F.relu(self.bn21(self.conv21(x1p)))
		x22 = F.relu(self.bn22(self.conv22(x21)))
		x2p, id2 = F.max_pool2d(x22,kernel_size=(1,2), stride=(1,2),return_indices=True)

		# Stage 3
		x31 = F.relu(self.bn31(self.conv31(x2p)))
		x32 = F.relu(self.bn32(self.conv32(x31)))
		x33 = F.relu(self.bn33(self.conv33(x32)))
		x3p, id3 = F.max_pool2d(x33,kernel_size=(1,2), stride=(1,2),return_indices=True)

		# # Stage 4
		# x41 = F.relu(self.bn41(self.conv41(x3p)))
		# x42 = F.relu(self.bn42(self.conv42(x41)))
		# x43 = F.relu(self.bn43(self.conv43(x42)))
		# x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

		# # Stage 5
		# x51 = F.relu(self.bn51(self.conv51(x4p)))
		# x52 = F.relu(self.bn52(self.conv52(x51)))
		# x53 = F.relu(self.bn53(self.conv53(x52)))
		# x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
		# print(x3p.shape)
		# x = nn.AvgPool2d(kernel_size = x3p.size()[1])(x3p)
		x = x3p.view(-1, x3p.size()[1]*x3p.size()[2]*x3p.size()[3])
		# print(x.shape)
		# exit()
		x_latent = self.fc1(x)
		x_lab = self.fc2(x)

		return id3, id2, id1, x_latent, x_lab	

class SegNet_dec(nn.Module):
	def __init__(self, out_channels):
		super(SegNet_dec, self).__init__()
		batchNorm_momentum = 0.1

		self.conv11 = nn.Conv2d(in_channels, 64, kernel_size=(1,3), padding=(0,1))
		self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
		self.conv12 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
		self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

		self.conv21 = nn.Conv2d(64, 128, kernel_size=(1,3), padding=(0,1))
		self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
		self.conv22 = nn.Conv2d(128, 128, kernel_size=(1,3), padding=(0,1))
		self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

		self.conv31 = nn.Conv2d(128, 256, kernel_size=(1,3), padding=(0,1))
		self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
		self.conv32 = nn.Conv2d(256, 256, kernel_size=(1,3), padding=(0,1))
		self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
		self.conv33 = nn.Conv2d(256, 256, kernel_size=(1,3), padding=(0,1))
		self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

		self.conv41 = nn.Conv2d(256, 512, kernel_size=(1,3), padding=(0,1))
		self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv42 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv43 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv51 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv52 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv53 = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv53d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv52d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv51d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

		self.conv43d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv42d = nn.Conv2d(512, 512, kernel_size=(1,3), padding=(0,1))
		self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
		self.conv41d = nn.Conv2d(512, 256, kernel_size=(1,3), padding=(0,1))
		self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

		self.conv33d = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1))
		self.bn33d = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
		self.conv32d = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1))
		self.bn32d = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
		self.conv31d = nn.Conv2d(32,  16, kernel_size=(1,3), padding=(0,1))
		self.bn31d = nn.BatchNorm2d(16, momentum= batchNorm_momentum)

		self.conv22d = nn.Conv2d(16, 16, kernel_size=(1,3), padding=(0,1))
		self.bn22d = nn.BatchNorm2d(16, momentum= batchNorm_momentum)
		self.conv21d = nn.Conv2d(16, 8, kernel_size=(1,3), padding=(0,1))
		self.bn21d = nn.BatchNorm2d(8, momentum= batchNorm_momentum)

		self.conv12d = nn.Conv2d(8, 8, kernel_size=(1,3), padding=(0,1))
		self.bn12d = nn.BatchNorm2d(8, momentum= batchNorm_momentum)
		self.conv11d = nn.Conv2d(8, in_channels, kernel_size=(1,3), padding=(0,1))

		self.fc1 = nn.Linear(in_features = num_classes + int(WL/16), out_features = int(WL/8), bias = True)
		self.convtr = nn.ConvTranspose2d(int(WL//8), 32, kernel_size = (1,int(WL//8)), stride=(1,1))


	def forward(self,id3, id2, id1, x_latent, x_lab):

		x = F.relu(self.fc1(torch.cat((x_lab, x_latent), 1)))
		# print(x.shape)
		# exit()
		x = x.view(-1,int(WL/8), 1, 1)
		# print(x.shape)
		# exit()
		x41d = F.relu(self.convtr(x))
		# Stage 5d
		# x5d = F.max_unpool2d(x5p, id5, kernel_size=(1,2), stride=(1,2))
		# x53d = F.relu(self.bn53d(self.conv53d(x5d)))
		# x52d = F.relu(self.bn52d(self.conv52d(x53d)))
		# x51d = F.relu(self.bn51d(self.conv51d(x52d)))

		# # Stage 4d
		# x4d = F.max_unpool2d(x51d, id4, kernel_size=(1,2), stride=(1,2))
		# x43d = F.relu(self.bn43d(self.conv43d(x4d)))
		# x42d = F.relu(self.bn42d(self.conv42d(x43d)))
		# x41d = F.relu(self.bn41d(self.conv41d(x42d)))

		# Stage 3d
		x3d = F.max_unpool2d(x41d, id3, kernel_size=(1,2), stride=(1,2))
		x33d = F.relu(self.bn33d(self.conv33d(x3d)))
		x32d = F.relu(self.bn32d(self.conv32d(x33d)))
		x31d = F.relu(self.bn31d(self.conv31d(x32d)))

		# Stage 2d
		x2d = F.max_unpool2d(x31d, id2, kernel_size=(1,2), stride=(1,2))
		x22d = F.relu(self.bn22d(self.conv22d(x2d)))
		x21d = F.relu(self.bn21d(self.conv21d(x22d)))

		# Stage 1d
		x1d = F.max_unpool2d(x21d, id1, kernel_size=(1,2), stride=(1,2))
		x12d = F.relu(self.bn12d(self.conv12d(x1d)))
		x11d = self.conv11d(x12d)

		# print(x.size())
		return x11d

class latent_discr(nn.Module):
	def __init__(self):
		super(latent_discr, self).__init__()
		self.fc1 = nn.Linear(in_features = int(WL/16), out_features = 16)
		self.fc2 = nn.Linear(in_features = 16, out_features = 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		# print(x.size())
		x = F.sigmoid(self.fc2(x))
		# print(x.size())
		return x

class cat_discr(nn.Module):
	def __init__(self):
		super(cat_discr, self).__init__()
		self.fc = nn.Linear(in_features = num_classes, out_features = 2, bias = True)

	def forward(self, x):
		# x = F.sigmoid(self.fc(x))
		x = self.fc(x.double())
		return(x)

# if str(sys.argv[1]) == 'extra_orig':
# 	filename = '../../cervical_cancer_data/train_all.zip'
# else:
# 	filename = '../../cervical_cancer_data/train.zip'
img_file_entries = []
img_labels = []
count_1 = 0
count_2 = 0
count_3 = 0
count = 0
# with ZipFile(filename) as archive:
# 	for idx,entry in enumerate(archive.infolist()):
# 		try:
# 			img = Image.open(archive.open(entry))
# 			img = img.convert('RGB')
# 			if 'Type_1/1339.jpg' in entry.filename and '.jpg' in entry.filename:
# 				continue
			
# 			elif '/Type_1' in entry.filename and '.jpg' in entry.filename:
# 				img_file_entries.append(entry)
# 				img_labels.append(0)
# 				count_1 = count_1 + 1
# 			elif '/Type_2' in entry.filename and '.jpg' in entry.filename:
# 				img_file_entries.append(entry)
# 				img_labels.append(1)
# 				count_2 = count_2+ 1
# 			elif '/Type_3' in entry.filename and '.jpg' in entry.filename:
# 				img_file_entries.append(entry)
# 				img_labels.append(2)
# 				count_3 = count_3 + 1
# 			count += 1
# 			print(count)
# 		except Exception as E:
# 			print(E)
# 			continue
		
#             with archive.open(entry) as file:
#                 img = Image.open(file)
#                 print(img.size, img.mode, len(img.getdata()))
# print(count_1)
# print(count_2)
# print(count_3)
# temp_img = []
# temp_labels = []
# for item in [0,1,2]:
# 	count=0
# 	for i in range(len(img_labels)):
# 		if img_labels[i] == item:
# 			temp_img.append(img_file_entries[i])
# 			temp_labels.append(img_labels[i])
# 			count += 1
# 		if count == 1300:
# 			break
# img_file_entries = temp_img
# img_labels = temp_labels
# In[3]:


class cervical_cancer(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, entries, labels, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
#         self.landmarks_frame = pd.read_csv(csv_file)
		
		# self.zipfile_path = zipfile_path
		self.entries = entries
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.entries)

	def __getitem__(self, idx):
		image = self.entries[idx]				
		image = image.reshape(1, np.shape(image)[0], np.shape(image)[1])
		image = torch.from_numpy(image)
				# print(np.max(np.array(image)))
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)

		return image, label


# In[4]:


def load_data(split, BatchSize):
	# apply_transform = transforms.Compose([transforms.Resize((net_ip_size, net_ip_size)),
	# 	transforms.RandomRotation(rot_deg),
	# 	transforms.RandomHorizontalFlip(),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 									 std=[0.229, 0.224, 0.225]),
	# ])
	# apply_val_transform = transforms.Compose([transforms.Resize((net_ip_size, net_ip_size)),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 									 std=[0.229, 0.224, 0.225]),
	# ])
	# apply_transform = transforms.Compose([transforms.ToTensor()])
	train_entries, val_entries, train_labels, val_labels = model_selection.train_test_split(img_file_entries, img_labels, test_size = val_split, random_state = 1)
	train_data = cervical_cancer(entries=train_entries, labels=train_labels)
	val_data = cervical_cancer(entries=val_entries, labels=val_labels)
	trainLoader = torch.utils.data.DataLoader(train_data, batch_size=BatchSize,
										  shuffle=True, num_workers=4) # Creating dataloader
	valLoader = torch.utils.data.DataLoader(val_data, batch_size=BatchSize,
										  shuffle=True, num_workers=4)
	return(train_data, val_data, trainLoader, valLoader)


in_channels = 1
out_channels = 1
# start_filter = 8
# in_disc = 16 * start_filter

# net = models.vgg16_bn(pretrained = True).features
encoder = SegNet_enc(in_channels)

# enc_children = []
# for child in encoder.named_children():
#     enc_children.append(child[0])

# net_children = []
# for child in net.named_children():
#     net_children.append(child[0])

# net_tr_ch = []
# for item in net_children:
#     if 'Conv2d' in str(net[int(item)]) or 'BatchNorm2d' in str(net[int(item)]):
#         net_tr_ch.append(item)
#         # print(item)
#         # print(net[int(item)])

# for i in range(len(net_tr_ch)):
#     getattr(encoder,enc_children[i]).parameters = getattr(net,net_tr_ch[i]).parameters        

decoder = SegNet_dec(out_channels)
# latent_disc = latent_discr()
cat_disc = cat_discr()

if sys.argv[2] == 'from_middle':
	encoder.load_state_dict(torch.load(str(sys.argv[1]) + '_encoder.pt', map_location='cpu'))
	print('1 down')
	decoder.load_state_dict(torch.load(str(sys.argv[1]) + '_decoder.pt', map_location='cpu'))
	print('2 down')
	# latent_disc.load_state_dict(torch.load('No_crop_equal_clswt_Adam_0.0001_pretr_enc_cat_disc_Seg_Net_' + str(sys.argv[1]) + '_latent_disc.pt', map_location='cpu'))
	cat_disc.load_state_dict(torch.load(str(sys.argv[1]) + '_cat_disc.pt', map_location='cpu'))
	print('3 down')
	print('Weights loaded')

encoder = encoder.double().cuda()
decoder = decoder.double().cuda()
# latent_disc = latent_disc.cuda()
cat_disc = cat_disc.double().cuda()
# exit()


# class_weights = compute_class_weight('balanced', np.unique(img_labels), img_labels)

# count_dict = Counter(img_labels)
# max_cls_freq = max(Counter(img_labels).values())
# class_weights = np.asarray([np.sqrt(float(max_cls_freq)/count_dict[label]) for label in np.unique(np.asarray(img_labels)).tolist()])

recon_loss = torch.nn.MSELoss()
# adversarial_loss = torch.nn.BCELoss()
adversarial_loss = torch.nn.CrossEntropyLoss()
cl_loss = torch.nn.CrossEntropyLoss()

optimizer_cl = torch.optim.Adam(encoder.parameters(), lr=0.0001)
optimizer_rec = torch.optim.Adam([{'params' : encoder.parameters()}, {'params' : decoder.parameters()}], lr=0.0001)
optimizer_gen = torch.optim.Adam(encoder.parameters(), lr=0.0001)
# optimizer_ld = torch.optim.Adam(latent_disc.parameters(), lr=0.0001)
optimizer_cd = torch.optim.Adam(cat_disc.parameters(), lr=0.0001)

#------------------------------ENCODER is the GENERATOR---------------------------------

# import warnings
# warnings.filterwarnings('ignore')

prev_time=os.stat('data.csv').st_mtime
prev_time_2 = os.stat('label.txt').st_mtime

temp = 0
while(1):
	print('Searching.....')
	while(os.stat('data.csv').st_mtime!=prev_time):
		
		#send to doctor
		print('data received')
		while os.stat('label.txt').st_mtime==prev_time_2:
			continue
		

		df = pd.read_csv('data.csv')
		with open('label.txt') as f:
			for line in f:
				line = line.strip()
		list_arr = []
		label_arr = []
		for i in range(int(df.shape[0]/WL)):
		 	list_arr.append(np.array(df.iloc[i*WL:(i+1)*WL,0]).reshape(1,-1))
		 	label_arr.append(int(line))
		# img_file_entries = np.load('train_all_images.npy').tolist()
		# img_labels = np.load('train_all_labels.npy').tolist()
		img_file_entries = list_arr
		img_labels = label_arr
		val_split = 0.2
		# net_ip_size = 224
		# rot_deg = 15
		BatchSize = 16
		train_data, val_data, trainLoader, valLoader = load_data(val_split, BatchSize)


		# Size of train and validation datasets
		print('No. of samples in train set: '+str(len(trainLoader.dataset)))
		print('No. of samples in validation set: '+str(len(valLoader.dataset)))
		train_rec_Loss = [] # List for saving main loss per epoch
		train_cl_Loss = []
		train_gen_Loss = []
		train_ld_Loss = []
		train_cd_Loss = []
		trainAcc = []
		train_cd_acc = []
		train_ld_acc = []
		 # List for saving training accuracy per epoch
		val_rec_Loss = [] # List for saving testing loss per epoc
		val_cl_Loss = []
		val_gen_Loss = []
		val_ld_Loss = []
		val_cd_Loss = []
		val_cd_acc = []
		val_ld_acc = []
		valAcc = []
		def train_and_val_net(epoch_nos):
			# for param in net.parameters():
			#     print(param.size())
			iterations = epoch_nos
			total_train_samples = float(len(trainLoader.dataset))
			total_val_samples = float(len(valLoader.dataset))
			max_val_Acc = 0
			best_dict = {}
			best_dict['epochs_till_now'] = 0

			start = time.time()
			for epoch in range(iterations):
				running_rec_Loss = 0.0 
				avg_rec_Loss = 0.0
				running_cl_Loss = 0.0 
				avg_cl_Loss = 0.0
				running_gen_Loss = 0.0
				avg_gen_Loss = 0.0
				running_ld_Loss = 0.0 
				avg_ld_Loss = 0.0
				running_cd_Loss = 0.0 
				avg_cd_Loss = 0.0
				running_correct = 0
				running_cd_correct = 0
				running_ld_correct = 0

				encoder.train(True) # For validating
				decoder.train(True)
				# latent_disc.train(True)
				cat_disc.train(True)
				idx=0
				for idx_1, data in enumerate(trainLoader):
					# print(idx)	
					inputs,labels = data
					# Adversarial ground truths
					valid = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
					fake = Variable(Tensor(inputs.shape[0]).fill_(0).long().cuda(), requires_grad=False)
					valid2 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
					fake2 = Variable(Tensor(inputs.shape[0]).fill_(0).long().cuda(), requires_grad=False)
					valid3 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
					valid4 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)

					real_imgs = Variable(inputs.double().cuda())#, requires_grad = True)
					labels = Variable(labels.long().cuda())

					# -----------------
					#  Train autoencoder for reconstruction
					# -----------------
					
					optimizer_rec.zero_grad()
					
					id3, id2, id1, x_latent, x_lab = encoder(real_imgs)
					out_imgs = decoder(id3, id2, id1, x_latent, F.softmax(x_lab))

					rec_loss = recon_loss(out_imgs, real_imgs)
					running_rec_Loss += rec_loss
					
					rec_loss.backward()
					optimizer_rec.step()
					# exit()


					# ---------------------
					#  Train Discriminator
					# ---------------------

					temp_cat = np.random.randint(low=0, high=num_classes, size=inputs.shape[0])
					real_cat = Variable(torch.from_numpy(np.eye(num_classes)[temp_cat]).float().cuda())
					# real_z = Variable(x_latent.data.new(x_latent.size()).normal_(0.0, 1.0).float().cuda())

					#LATENT DISCRIMINATOR STARTS

					# optimizer_ld.zero_grad()
					# # Measure discriminator's ability to classify real from generated samples
					# # out_real = torch.mean(discriminator(real_imgs))
					# # out_gen = torch.mean(discriminator(gen_imgs.detach()))
					# # print('\n')
					# # print(out_real)
					# # print(out_gen)
					# # print('\n')

					# _,_,_,_,_, x_latent, _ = encoder(real_imgs)
					# comp_fake = latent_disc(x_latent)
					# comp_real =  latent_disc(real_z)
					# fake_z_loss = adversarial_loss(comp_fake, fake2)
					# real_z_loss = adversarial_loss(comp_real, valid2)
					# ld_loss = (real_z_loss + fake_z_loss)
					# for i in range(comp_fake.size()[0]):
					# 	if comp_fake[i] < 0.5 and comp_real[i] > 0.5:
					# 		running_ld_correct += 1
					# running_ld_Loss += ld_loss
					# ld_loss.backward()
					# optimizer_ld.step()		

					#LATENT DISCRIMINATOR STARTS

					# optimizer_gen.zero_grad()
					# _, _, _, _, _, x_latent, x_lab = encoder(real_imgs)
					# gen_z_loss = adversarial_loss(latent_disc(x_latent), valid3)
					# gen_c_loss = adversarial_loss(cat_disc(F.softmax(x_lab)), valid4)
					# gen_loss = gen_z_loss + gen_c_loss
					# running_gen_Loss += gen_loss
					# gen_loss.backward()
					# optimizer_gen.step()

					
					optimizer_cd.zero_grad()	
					_,_, _, _, x_lab = encoder(real_imgs)

					# running_cd_Loss += cd_loss
					# running_ld_Loss += ld_loss
					comp_fake = cat_disc(F.softmax(x_lab))
					comp_real =  cat_disc(real_cat)


					fake_c_loss = adversarial_loss(comp_fake, fake)
					real_c_loss = adversarial_loss(comp_real, valid)
					cd_loss = (real_c_loss + fake_c_loss)
					for i in range(valid.size()[0]):
						if comp_real[i][1] > comp_fake[i][1]:
							running_cd_correct += 1
					running_cd_Loss += cd_loss
					
					cd_loss.backward()
					optimizer_cd.step()



					# -----------------
					#  Train generator (or encoder)
					# -----------------
					
					optimizer_gen.zero_grad()
					_, _, _, _, x_lab = encoder(real_imgs)
					# gen_z_loss = adversarial_loss(latent_disc(x_latent), valid3)
					gen_c_loss = adversarial_loss(cat_disc(F.softmax(x_lab)), valid4)
					# gen_loss = gen_z_loss + gen_c_loss
					gen_loss = gen_c_loss
					running_gen_Loss += gen_loss
					
					gen_loss.backward()
					optimizer_gen.step()
					
					optimizer_cl.zero_grad()
					_,_,_,_,x_lab = encoder(real_imgs)
					# temp_lab = labels.cpu().numpy().reshape((-1))
					# act_lab = Variable(torch.from_numpy(np.eye(3)[temp_lab]).cuda())
					# g_loss = -adversarial_loss(discriminator(gen_z), fake)
					class_loss = cl_loss(x_lab, labels.long())
					_, predicted = torch.max(x_lab.data, 1)
					running_correct += (predicted.cpu() == labels.data.cpu()).sum()
					# real_imgs.requires_grad = False
					running_cl_Loss += class_loss
					
					class_loss.backward()
					optimizer_cl.step()
					idx=idx+1
					# print(idx)

				avg_rec_Loss = running_rec_Loss/float(idx)
				avg_cl_Loss = running_cl_Loss/float(idx)
				avg_gen_Loss = running_gen_Loss/float(idx)
				avg_ld_Loss = running_ld_Loss/float(idx)
				avg_cd_Loss = running_cd_Loss/float(idx)
				avg_trainacc = float(running_correct)/total_train_samples
				avg_ld_acc = float(running_ld_correct)/total_train_samples
				avg_cd_acc = float(running_cd_correct)/total_train_samples

				train_rec_Loss.append(avg_rec_Loss)
				train_cl_Loss.append(avg_cl_Loss)
				train_gen_Loss.append(avg_gen_Loss)
				train_ld_Loss.append(avg_ld_Loss)
				train_cd_Loss.append(avg_cd_Loss)
				trainAcc.append(avg_trainacc)
				train_ld_acc.append(avg_ld_acc)
				train_cd_acc.append(avg_cd_acc)

				# if epoch > 0:
				# 	generator.load_state_dict(g_state_dict)
				# 	discriminator.load_state_dict(d_state_dict)

				# for param in encoder.parameters():
				# 	param.requires_grad = True
				# for param in decoder.parameters():
				# 	param.requires_grad = True
				# for param in discriminator.parameters():
				# 	param.requires_grad = True



				encoder.eval() # For training
				decoder.eval()
				# latent_disc.eval()
				cat_disc.eval()

				print('---------------------After training----------------------')

				running_rec_Loss = 0.0 
				
				running_cl_Loss = 0.0

				running_gen_Loss = 0.0 
				
				running_ld_Loss = 0.0 

				running_cd_Loss = 0.0

				running_correct = 0

				running_cd_correct = 0

				running_ld_correct = 0


				with torch.no_grad():
					idx=0
					for idx_1, data in enumerate(valLoader):
						inputs,labels = data
						# Adversarial ground truths
						valid = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
						fake = Variable(Tensor(inputs.shape[0]).fill_(0).long().cuda(), requires_grad=False)
						valid2 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
						fake2 = Variable(Tensor(inputs.shape[0]).fill_(0).long().cuda(), requires_grad=False)
						valid3 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)
						valid4 = Variable(Tensor(inputs.shape[0]).fill_(1).long().cuda(), requires_grad=False)

						real_imgs = Variable(inputs.double().cuda())#, requires_grad = True)
						labels = Variable(labels.long().cuda())

						# -----------------
						#  Train autoencoder for reconstruction
						# -----------------

						# optimizer_rec.zero_grad()
						
						id3, id2, id1, x_latent, x_lab = encoder(real_imgs)
						out_imgs = decoder(id3, id2, id1, x_latent, F.softmax(x_lab))

						rec_loss = recon_loss(out_imgs, real_imgs)
						running_rec_Loss += rec_loss
						# rec_loss.backward()
						# optimizer_rec.step()
						# exit()


						# ---------------------
						#  Train Discriminator
						# ---------------------

						temp_cat = np.random.randint(low=0, high=num_classes, size=inputs.shape[0])
						real_cat = Variable(torch.from_numpy(np.eye(num_classes)[temp_cat]).float().cuda())
						# real_z = Variable(x_latent.data.new(x_latent.size()).normal_(0.0, 1.0).float().cuda())

						# optimizer_ld.zero_grad()
						# Measure discriminator's ability to classify real from generated samples
						# out_real = torch.mean(discriminator(real_imgs))
						# out_gen = torch.mean(discriminator(gen_imgs.detach()))
						# print('\n')
						# print(out_real)
						# print(out_gen)
						# print('\n')

						# _,_,_,_,_, x_latent, _ = encoder(real_imgs)
						# comp_fake = latent_disc(x_latent)
						# comp_real = latent_disc(real_z)
						# fake_z_loss = adversarial_loss(comp_fake, fake2)
						# real_z_loss = adversarial_loss(comp_real, valid2)
						# ld_loss = (real_z_loss + fake_z_loss)
						# running_ld_Loss += ld_loss
						# for i in range(comp_fake.size()[0]):
						# 	if comp_fake[i] < 0.5 and comp_real[i] > 0.5:
						# 		running_ld_correct += 1
						# ld_loss.backward()
						# optimizer_ld.step()		

						# optimizer_cd.zero_grad()	

						# _,_,_,_, _,_, x_lab = encoder(real_imgs)

						# running_cd_Loss += cd_loss
						# running_ld_Loss += ld_loss
						comp_fake = cat_disc(F.softmax(x_lab))
						comp_real = cat_disc(real_cat)
						fake_c_loss = adversarial_loss(comp_fake, fake)
						real_c_loss = adversarial_loss(comp_real, valid)
						cd_loss = (real_c_loss + fake_c_loss)
						running_cd_Loss += cd_loss
						for i in range(valid.size()[0]):
							if comp_real[i][1] > comp_fake[i][1]:
								running_cd_correct += 1
						# cd_loss.backward()
						# optimizer_cd.step()



						# -----------------
						#  Train generator (or encoder)
						# -----------------

						# optimizer_gen.zero_grad()
						# _, _, _, _, _,x_latent, x_lab = encoder(real_imgs)
						# gen_z_loss = adversarial_loss(latent_disc(x_latent), valid3)
						gen_c_loss = adversarial_loss(cat_disc(F.softmax(x_lab)), valid4)
						gen_loss = gen_c_loss
						running_gen_Loss += gen_loss
						# gen_loss.backward()
						# optimizer_gen.step()

						# optimizer_cl.zero_grad()
						# _,_,_,_,_,_,x_lab = encoder(real_imgs)
						_, predicted = torch.max(x_lab.data, 1)
						running_correct += (predicted.cpu() == labels.data.cpu()).sum()
						# temp_lab = labels.cpu().numpy().reshape((-1))
						# act_lab = Variable(torch.from_numpy(np.eye(3)[temp_lab]).cuda())
						# g_loss = -adversarial_loss(discriminator(gen_z), fake)
						class_loss = cl_loss(x_lab, labels)
						# real_imgs.requires_grad = False
						running_cl_Loss += class_loss
						# class_loss.backward()
						# optimizer_cl.step()
						# print(idx)
						if idx == 0:
							
							totalPreds = predicted                    
							tempLabels = labels.data.cpu()
												
						else:
							
							totalPreds = torch.cat((totalPreds,predicted),0)                 
							tempLabels = torch.cat((tempLabels,labels.data.cpu()),0)
						idx = idx + 1
					

				avg_rec_val_Loss = running_rec_Loss/float(idx)
				avg_cl_val_Loss = running_cl_Loss/float(idx)
				avg_gen_val_Loss = running_gen_Loss/float(idx)
				# avg_ld_val_Loss = running_ld_Loss/float(idx)
				avg_cd_val_Loss = running_cd_Loss/float(idx)
				avg_val_acc = float(running_correct)/total_val_samples
				# avg_val_ld_acc = float(running_ld_correct)/total_val_samples
				avg_val_cd_acc = float(running_cd_correct)/total_val_samples		

				val_rec_Loss.append(avg_rec_val_Loss)
				val_cl_Loss.append(avg_cl_val_Loss)
				val_gen_Loss.append(avg_gen_val_Loss)
				# val_ld_Loss.append(avg_ld_val_Loss)
				val_cd_Loss.append(avg_cd_val_Loss)
				valAcc.append(avg_val_acc)
				# val_ld_acc.append(avg_val_ld_acc)
				val_cd_acc.append(avg_val_cd_acc)

				conf_arr = confusion_matrix(tempLabels.cpu().numpy(), totalPreds.cpu().numpy())

				print('Iteration: {:.0f} /{:.0f} Model ; time: {:.3f} secs'.format(epoch + 1,iterations, (time.time() - start)))
				print('Rec train Loss: {:.6f} '.format(avg_rec_Loss))
				print('Classifier train Loss: {:.6f} '.format(avg_cl_Loss))
				print('Rec val Loss: {:.6f} '.format(avg_rec_val_Loss))
				print('Classifier val Loss: {:.6f} '.format(avg_cl_val_Loss))
				print('Train Acc: {:.3f} '.format(np.mean(np.array(trainAcc))*100))
				# print('Train ld Acc: {:.3f}'.format(avg_ld_acc*100))
				print('Train cd Acc: {:.3f}'.format(avg_cd_acc*100))		
				print('Validation Acc: {:.3f} '.format(np.mean(np.array(valAcc)*100)))
				# print('val ld Acc: {:.3f}'.format(avg_val_ld_acc*100))
				print('val cd Acc: {:.3f}'.format(avg_val_cd_acc*100))
				# print(conf_arr)
			# encoder_dict = {'inc':generator.inc.state_dict(), 'down1':generator.down1.state_dict(), 'down2':generator.down2.state_dict(), 'down3':generator.down3.state_dict(), 'down4':generator.down4.state_dict()}
				if avg_val_acc > max_val_Acc:
					max_val_Acc = avg_val_acc
					best_dict = {'best_epoch':epoch + 1, 'acc':max_val_Acc, 'tot_epoch':iterations, 'epochs_till_now':epoch + 1}
					torch.save(encoder.state_dict(), str(sys.argv[1]) + '_encoder_best_from.pt')			
					torch.save(best_dict, str(sys.argv[1]) + '_best_from.tar')			
				else:
					best_dict['epochs_till_now'] = epoch + 1
					torch.save(best_dict, str(sys.argv[1]) + '_best_from.tar')			
				# last_dict = {'rec_train_loss':train_rec_Loss, 'cl_train_loss':train_cl_Loss, 'gen_train_loss':train_gen_Loss, 'ld_train_loss':train_ld_Loss, 'cd_train_loss':train_cd_Loss, 'rec_val_loss':val_rec_Loss, 'cl_val_loss':val_cl_Loss, 'gen_val_loss':val_gen_Loss, 'ld_val_loss':val_ld_Loss, 'cd_val_loss':val_cd_Loss, 'trainAcc':trainAcc, 'train_ld_acc':train_ld_acc, 'train_cd_acc':train_cd_acc, 'valAcc':valAcc, 'val_ld_acc':val_ld_acc, 'val_cd_acc':val_cd_acc, 'tot_epoch':iterations}
				last_dict = {'rec_train_loss':train_rec_Loss, 'cl_train_loss':train_cl_Loss, 'gen_train_loss':train_gen_Loss, 'cd_train_loss':train_cd_Loss, 'rec_val_loss':val_rec_Loss, 'cl_val_loss':val_cl_Loss, 'gen_val_loss':val_gen_Loss, 'cd_val_loss':val_cd_Loss, 'trainAcc':trainAcc, 'train_cd_acc':train_cd_acc, 'valAcc':valAcc, 'val_cd_acc':val_cd_acc, 'tot_epoch':iterations}

				torch.save(last_dict, str(sys.argv[1]) + '_graph_from.tar')

				torch.save(encoder.state_dict(),str(sys.argv[1]) + '_encoder.pt')
				torch.save(decoder.state_dict(), str(sys.argv[1]) + '_decoder.pt')
				# torch.save(latent_disc.state_dict(),'No_crop_equal_clswt_Adam_0.0001_pretr_enc_cat_disc_Seg_Net_' + str(sys.argv[1]) + '_latent_disc.pt')
				torch.save(cat_disc.state_dict(), str(sys.argv[1]) + '_cat_disc.pt')

			if(np.mean(valAcc) > 0.5):
				send = int(line)
			else:
				send = 1-int(line)
			if send == 0:
				text = 'No action required'
			else:
				text = 'Please take suitable action.'
			return(text)
		 # List for saving testing accuracy per epoch
			


		text = train_and_val_net(10)
		with open('send.txt', 'w') as f:
			f.write(text)
		#os.system('sshpass -p "%s" scp "%s" "%s:%s"' % ("peace@123","./send.txt", "gopa@10.10.1.122", "/home/gopa/anom_frames"))
		print('Sending to client')
		time.sleep(1)
		print('sent')
		prev_time = os.stat('data.csv').st_mtime
		prev_time_2 = os.stat('label.txt').st_mtime
		break