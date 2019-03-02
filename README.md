# FaultyWaterPumps-Prediction
This is an ML project from one of the Drivendata.org datasets, whose target is to classify a given water pump's functionality into either Functional, Not functional, Functional but needs repair



Pre-requirement on machine:
- Python 3.6
- Scikit-learn, pandas, numpy packages installed

Executing and running the project:

Folder structure as submitted :

Root(ML-project)
|
---Code--|
|	 ---Models
|	 ---Preprocessing
|
|
|
---Datasets--|
|	     |
|	     ---Training_set_labels.csv
|	     ---Training_set_values.csv
|	     ---test_values_processed.csv
|	     ---dataset_processed.csv
---Readme.txt
|
---Final_report.pdf



1. To run the code:
	
	Preprocessing:Since we are displays plots in this, you cannot run it in command line/terminal.
	              So recommended to run it in Jupyter using the prepro.py file inside the Preprocessing folder

	 	
	Running Models:
	Go inside Models folder and run the below command :
	
		Syntax: python Modelname.py

		Ex: python adaboost.py
	    	    python SVC.py
	    	  
		After running above command in command line you will see 
			-->Classification report, 
			-->Accuracy of the model using train-test split  
			
 
	
