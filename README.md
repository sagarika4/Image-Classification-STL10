# Image-Classification-STL10
Supervised Image classification done on [STL10](https://cs.stanford.edu/~acoates/stl10/) dataset using [transfer learning](https://cs231n.github.io/transfer-learning/). I have used the pretrained [ResNet18](https://arxiv.org/abs/1512.03385) model
for this task.

Deep Learning Framework used: Pytorch

### Data Loading:
- Dataset Divsion:
   * Used predefined training set consisting of 5000 labelled image. The unlabelled dataset has not been used to training the model. 
   * The validation and test sets have been split in 40:60 ratio from the predeifned test set of the dataset
- Data Augmentation using Image Transformations on the training set:
  1. Padding
  2. Random cropping followed by resizing
  3. Rotate by 10 degrees
  4. Random Horizontal Flip with probablilty 0.5
  5. [Cutout Transformation](https://github.com/uoguelph-mlrg/Cutout)
- Loaded the data in batches of 256 images each
  
### Transfer Learning
  - Estimated an optimal Learning Rate:
    * Used the approach in Section 3.3 of [Cyclical Learning Rates For Training Neural Networks](https://doi.org/10.1007/978-3-319-97982-3_16) to find an optimal learning rate
    * Used [pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) to implement the above.
  - Used mini-batch gradient descent on a batch size of 256 images.
  - Trained the architecture using the holdout validation technique. The model weights were selected based on highest validation accuracy.
  - Fine Tuned Model
    * Achieved the Test Accuracy on 4800 images of about 96%.
  - ConvNet as Fixed Feature exractor:
    * Achieved the Test Accuracy on 4800 images of about 93%.
    
 ### [Class Activation Mapping(CAM)](http://cnnlocalization.csail.mit.edu/)
  - CAM can be utilised to analyse why the network made a particular decision
  - I used CAM to get the feature mappings from the last convolutional layer of the network since this layer will contain the most information regarding detected patterns, localised in space.
  - The heatmaps generated from CAM can be superimposed on the original image to understand the reasonings behing the model's decision.
  
  You can view the transfer_learning.ipynb using nbviewer [here](https://nbviewer.jupyter.org/github/sagarika4/Image-Classification-STL10/blob/master/transfer_learning_STL10.ipynb)
   


