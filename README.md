# FinalYearProject
Repo for keeping track of my final year project for GY350(Computer Science &amp; Information Technology) titled "Using Machine Learning to Detect Alzheimer's Disease" supervised by Dr.Frank Glavin 

I will be creating a machine learning model that will be trained on a dataset of brain MRIs that are seperated into 4 classes ("non-demented", "very-mild demented", "mild demented", "moderate demented"). The model will be hosted in docker and will be able to take an input image from a user and will return a classification and heatmap highlighting the most influential components of the input image to the model's decision. 

It will involve using an ensemble model of VGG-16 and EfficientNet-B2. I will implement Convolutional Attention Block Model(CBAM) to improve accuracy. I will also implement Gradient-weighted Class Activation Mapping (Grad-CAM) which is an explainable AI to produce the heatmap that will be displayed to the user. 

The project also involves data augmentation and producing synthetic data as the dataset I am using suffers from the common problem in this space of being an inbalanced dataset as it is rarer for moderately demented patients to have MRIs taken of their brains. 

link to final report: https://docs.google.com/document/d/1dyUyXSf_fjxKJT3lI9qAXNC8E-IDvtXi8qhIhwMfgaA/edit?tab=t.0
