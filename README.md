# Disaster Response Pipeline Project

# Objective
This project aims to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. 

There is a data set containing real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events in order to send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Files:
- data:
    - disaster_categories.csv: contains all data about how each message is categorized;
    - disaster_messages.csv: contains all messages sent.
    - process_data.py: Process all data from data sets and creates MySQLdb.db
    - MySQLdb.db: Database created after processing all data and ready to be pass through the model.

- models:
    - train_classifier.py: Teste data from MySQLdb.db and determines the best paremeters for a randomforest classifier. Exports MyModel.pkl.
    - MyModel.pkl: Model used to classify messages inputed in front end.  

- app:
    - templates: contains all html to run web app.
    - run.py: runs the app and connect frontend to model. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv "sqlite:///MySQLdb.db"'
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py "sqlite:///../data/MySQLdb.db" "MyModel.pkl"`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
