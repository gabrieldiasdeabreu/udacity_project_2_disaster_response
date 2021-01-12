# Disaster Response Pipeline Project

## Images

![main page](https://github.com/gabrieldiasdeabreu/udacity_project_2_disaster_response/blob/master/mainPage.png?raw=true)
![message classification page](https://github.com/gabrieldiasdeabreu/udacity_project_2_disaster_response/blob/master/messageClassificationPage.png?raw=true)

## SETUP
`pyenv shell 3.7`
`python -m venv venv`
`python install -r requirements.txt`

## Model
This project applies an optimized random forest to the disaster pipeline problem. 
The data has been processed to each of 36 classes being an binary feature which indicates if the text is one of those categories when 1 and 0 otherwise.

Some categories in the dataset have much less examples than others as we can see at the Overview. It decreases model learning generalization by biasing to more commom categories as related, aid_related and weather_related.

## Project
```bash
.
├── app
│   ├── run.py - runs the flask server
│   └── templates - html files
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv - Categories csv data from disaster pipeline problem
│   ├── disaster_messages.csv - Messages csv data from disaster pipeline problem
│   ├── DisasterResponse.db - generated sqlite database from process_data step
│   └── process_data.py - ETL script to generate DisasterResponde.db from the csvs
├── models
│   ├── classifier.pkl - generated model from train_classifier script
│   └── train_classifier.py - script to generate classification models from DisasterResponse.db
├── README.md - this file
└── requirements.txt - dependencies file
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
