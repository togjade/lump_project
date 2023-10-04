
Description of the scripts:
    
    * config.yaml -> provides the parameters to train the SL model. 
    * choose_participants.ipynb -> helps to divide the dataset to train data and test data by taking out 2 out of 10 participants' data. The data of 2 participants is used as unseen test data to test the trained model's performance and generalizability. 
    * stratify_data_by_prototypes -> to train and test the models by each subclass.
    * train_dev_test_splitter.py -> prepare the test, train and dev (validation set) for the model training. 
    * utils.py -> train the model.
    *models.py -> defien the models.
