import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images_grid(inp, title, fontsize=30):
	""" Visualise a batch of images in form of grids along with the true/predicted label of each image."""

	inp = inp.numpy().transpose((1, 2, 0))  # Permute the dimensions of image to create plot
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.figure(figsize=[100, 100])
	plt.imshow(inp)
	plt.title(title, fontsize=fontsize)
	plt.pause(0.001)

def prepare_and_get_misclassified_images_data(incorrect_images, pred_classes_incorrect, pred_classes_correct):
	

	incorrect_images_visualise = torch.cat((incorrect_images[0], incorrect_images[1]), 0)

	# Make it a multiple of 10 for proper grid view
	total_images_display = 10 * int(incorrect_images_visualise.shape[0]/10) 
	incorrect_images_visualise = incorrect_images_visualise[0 : total_images_display]
	pred_classes_incorrect_display = torch.cat((pred_classes_incorrect[0], pred_classes_incorrect[1]), 0)[0 : total_images_display]
	pred_classes_correct_display = torch.cat((pred_classes_correct[0], pred_classes_correct[1]), 0)[0 : total_images_display]

	return incorrect_images_visualise, pred_classes_incorrect_display, pred_classes_correct_display

def visualise_images_misclassified(incorrect_images_visualise, pred_classes_incorrect_display, pred_classes_correct_display, class_names):

	plt.figure(figsize=[100, 100]) 
	out = torchvision.utils.make_grid(incorrect_images_visualise, nrow=10).cpu()
	title_incorrect_labels = [class_names[x] for x in pred_classes_incorrect_display]
	title_correct_labels = [class_names[x] for x in pred_classes_correct_display]
	title = "Predicted: " + str(title_incorrect_labels) + "\n\n" + "Actual: " + str(title_correct_labels)
  
	show_images_grid(out, title, 50)

def plot_accuracies(num_epochs, train_accuracy, validation_accuracy):
	""" Plots training and validation accuracies with respect to the number of epochs """

	fig, ax = plt.subplots()
	plt.grid()

	x=np.linspace(1, num_epochs, num_epochs).astype(int)
	ax.plot(x, train_accuracy, marker='o', markerfacecolor='red', color='blue',markersize=5,linewidth=4, label = "train accuracy")
	ax.plot(x, validation_accuracy, marker='o', markerfacecolor='red', color='orange',markersize=5,linewidth=4, label = "validation accuracy")
	ax.set_xlabel("#Epochs")
	ax.set_ylabel("Accuracy")
	ax.set_title("Train V/S Test Accuracy")

	plt.legend()
	plt.show()


class SaveFeatures():
	"""Saves the intermediate feature map for the target layer provided.
	Args:
		module (torch.nn.Module): The module of the model of which convolutional layer is to be extracted.
	"""

	features = None

	def __init__(self, m): 
		self.hook = m.register_forward_hook(self.hook_fn)

	def hook_fn(self, module, input, output): 
		self.features = ((output.cpu()).data).numpy()
	  
	def remove(self): 
		self.hook.remove()

def get_CAM_inputs(model, incorrect_images_visualise, pred_classes_correct_display):
	"""
	Returns the inputs required to create Class Activation Maps(CAM):
		a. The last convolutional layer of network before the fully connected layer
		b. The weights of fully connected layer.
		c. numpy array of indices of predicted and actual classes
	"""


	final_layer = model._modules.get('layer4')
	activated_features = SaveFeatures(final_layer)

	prediction = model(incorrect_images_visualise)
	pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
	
	activated_features.remove()

	total_images_display = incorrect_images_visualise.shape[0]
	class_idx_incorrect = torch.topk(pred_probabilities,1)[1].int().cpu().numpy()
	class_idx_correct = pred_classes_correct_display.cpu().numpy().reshape((total_images_display, 1))

	weight_softmax_params = list(model._modules.get('fc').parameters())
	weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

	return activated_features.features, class_idx_incorrect, class_idx_correct, weight_softmax

def get_CAM(feature_conv, weight_fc, class_idx):
	"""
	Create and get CAM by multiplying convolutional layer with the weights for the required class.
	"""
	num_data, nc, h, w = feature_conv.shape
	cam = np.einsum('ij,ijk->ik', weight_fc[class_idx].reshape((num_data, nc)), feature_conv.reshape((num_data, nc, h*w))) # num_data X h*w
	cam = cam - np.min(cam, 1).reshape((cam.shape[0], 1))
	cam = cam / np.max(cam, 1).reshape((cam.shape[0], 1))
	cam = cam.reshape(num_data, h, w)
	return cam

def get_CAM_on_image(image, mask):
	"""
	Create heatmap from the generated CAM and superimpose on the actual image. The heatmap will highlight the regions the network used for predicting the particular class for which CAM is generated.
	"""

	upsample = cv2.resize(np.uint8(mask * 255), (256,256))
	heatmap = cv2.applyColorMap(cv2.resize(upsample, image.shape[1:3]), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	image = image.cpu().numpy().transpose(1,2,0) # 224 X 224 X 3
	result = heatmap * 0.5 + image * 0.5
	return result


