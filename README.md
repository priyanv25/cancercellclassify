# cancercellclassify
The project demonstrates deployement of machine learning model using flask api.

Description - Classification of cancer cells as malignant/benign by taking various parameters such as marginal adhesion,cell size,bare nucleoli using SVM classifier. 

The project contains :

app.py- Flask APIs that receives details through GUI or API calls, computes the precited value based on the model and returns it.

model.py- Svm model used for cell classification.
templates- html/css templates used for user interface.
