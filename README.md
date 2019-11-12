# ImageClassifier
A Generic multi-class Image Classifier that can be re-purposed and retrained for a large number of use cases.
The example model builds on the ImageNet model to train data. It can be modified to use other models as well.

This is a general purpose image classifier that can be trained on any image set.

Steps to test the code
1. Add all images to the train set. Each image should be added under a specific category.
2. Run TrainAndSaveModel.py. It will train the model on the given data set. All arguments are optional.
  a. Specify the number of epochs to run for with the -e command
  b. Specify the batch_size with the -b command.
  c. Specify the location of image directories with the -t command.
  d. Specify the name and path of output model with the -m command.
  e. Specify the name and path of output categories file with the -c command.
  
 3. Once complete, you'll have two new files. Model and Categories; these are required to predict the new images.
 4. Run Predict.py. It will classify all images in the test path. All arguments are optional.
  a. Specify the location of test images with the -t command.
  b. Specify the model file to be used with the -m command.
  c. Specify the category file to be used with the -c command.
 5. The program will print the result on screen.
 
 Note: For best results, use atleast 1k images for each category.

How to: Adding images to Train/test set

TRAIN IMAGES
1. Create a dataset directory to place all training images in.
2. Create as many sub-folders as categories in your taining data inside the dataset directory.
2. Add images of a particular category to its respective subfolder.

Structure Example:

	TrainSet
	|- Sub Category 1
		|- Image 1.1
		|- Image 1.2
		|- Image 1.n
	|- Sub Category 2
		|- Image 2.1
		|- Image 2.2
		|- Image 2.n
	|- Sub category n
		|- Image n.1
		|- Image n.2
		|- Image n.n
	
TEST IMAGES
1. Place one or more images whose category is unknown to the test Images directory.

Structure Example:

	TestSet
	|- Unknown Image 1
	|- Unknown Image 2
	|- Unknown Image n
	
