import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torch.utils.data import SubsetRandomSampler
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

class Cutout(object):
	"""Randomly mask out one or more patches from an image.
	Args:
		n_holes (int): Number of patches to cut out of each image.
		length (int): The length (in pixels) of each square patch.
	"""

	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length

	def __call__(self, img):
		"""
		Args:
			img (Tensor): Tensor image of size (C, H, W).
		Returns:
			Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		h = img.size(1)
		w = img.size(2)

		mask = np.ones((h, w), np.float32)

		for n in range(self.n_holes):
			y = np.random.randint(h)
			x = np.random.randint(w)

			y1 = np.clip(y - self.length // 2, 0, h)
			y2 = np.clip(y + self.length // 2, 0, h)
			x1 = np.clip(x - self.length // 2, 0, w)
			x2 = np.clip(x + self.length // 2, 0, w)

			mask[y1: y2, x1: x2] = 0.

		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img = img * mask

		return img

def load_dataset(root_path, num_workers, batch_size, cutout_transformation=False, cutout_transformation_length=None):
	""" Load STL10 datasets into train, validation and test sets. Since the testing data > training data, the validation set is cut out from testing set with 40% split."""

	# Create test and validation sets from STL10 test split
	num_test = 8000
	indices = list(range(num_test))
	np.random.shuffle(indices)
	split = int(np.float(0.4 * num_test))
	test_idx, valid_idx = indices[split:], indices[:split]
	test_sampler = SubsetRandomSampler(test_idx)  # Randomly sample images from test indices
	valid_sampler = SubsetRandomSampler(valid_idx)  # Randomly sample images from validation indices

	# Image Transformations
	data_transforms = {
		'train': transforms.Compose([
			transforms.Pad(4),
			transforms.RandomCrop(96),
			transforms.Resize(224),
			transforms.RandomRotation(10),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'eval': transforms.Compose([
			transforms.Resize(244),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}

	if cutout_transformation:
		data_transforms['train'].transforms.append(Cutout(n_holes=1, length=cutout_transformation_length))

	train_loader = torch.utils.data.DataLoader(
		datasets.STL10(
			root=root_path, download=True, split='train',
			transform=data_transforms['train']),
		batch_size=batch_size, num_workers=num_workers, shuffle=True)
	val_loader = torch.utils.data.DataLoader(
		datasets.STL10(
			root=root_path, download=False, split='test',
			transform=data_transforms['eval']),
		batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
	test_loader = torch.utils.data.DataLoader(
		datasets.STL10(
			root=root_path, split='test', download=False,
			transform=data_transforms['eval']),
		batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

	dataloaders = {
		'train': train_loader,
		'val': val_loader,
		'test': test_loader
	}
	return dataloaders

def train_model(device, dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
	""" Trains the model using training set. The best weights for model are chosen on the basis of validation set. """

	dataset_sizes = {
		'train': len(dataloaders['train'].sampler),
		'val': len(dataloaders['val'].sampler)
	}

	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	train_accuracy = []
	validation_accuracy = []

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			if phase == 'train':
				train_accuracy.append(epoch_acc)
			if phase == 'val':
				validation_accuracy.append(epoch_acc)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc * 100))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, train_accuracy, validation_accuracy

def test(device, dataloaders, model):
	""" Tests the image classification on testing set."""

	test_acc = 0.0
	model.eval()  # Set the model to eval mode
	incorrect_images = []
	pred_classes_incorrect = []
	pred_classes_correct = []

	for inputs, labels in dataloaders['test']:
		with torch.no_grad():
			inputs, labels = inputs.to(device), labels.to(device)
			output = model(inputs)
			_, pred = torch.max(output, dim=1)
			correct = pred == labels
			mask = (pred != labels).view(-1)
			incorrect_images.append(inputs[mask])
			pred_classes_incorrect.append(pred[mask])
			pred_classes_correct.append(labels[mask])
			test_acc += torch.mean(correct.float())

	return test_acc, incorrect_images, pred_classes_incorrect, pred_classes_correct