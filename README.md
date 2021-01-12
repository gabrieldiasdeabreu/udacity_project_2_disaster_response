# Disaster Response Pipeline Project

## Sumary
This project aims to identify given a text message from which class of disaster it belongs and therefore how to help. Messages from the train dataset come from many sources as news, direct or social media and are provided by Figure Eight.

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

'''
Test model results:
                        precision    recall  f1-score   support
               related       0.97      0.99      0.98      4021
               request       0.98      0.89      0.93       943
                 offer       1.00      0.73      0.84        26
           aid_related       0.95      0.93      0.94      2224
          medical_help       0.99      0.80      0.89       436
      medical_products       1.00      0.81      0.89       286
     search_and_rescue       0.99      0.75      0.85       139
              security       1.00      0.81      0.90        97
              military       1.00      0.80      0.89       194
           child_alone       0.00      0.00      0.00         0
                 water       0.99      0.89      0.94       336
                  food       0.98      0.91      0.94       581
               shelter       0.99      0.85      0.91       459
              clothing       1.00      0.83      0.90        92
                 money       1.00      0.73      0.85       113
        missing_people       1.00      0.84      0.91        67
              refugees       0.98      0.76      0.86       168
                 death       0.99      0.78      0.88       254
             other_aid       0.98      0.81      0.89       715
infrastructure_related       0.99      0.78      0.87       338
             transport       1.00      0.79      0.88       259
             buildings       0.99      0.81      0.89       245
           electricity       0.99      0.79      0.88       116
                 tools       1.00      0.93      0.97        30
             hospitals       1.00      0.66      0.80        68
                 shops       1.00      0.78      0.88        23
           aid_centers       1.00      0.80      0.89        55
  other_infrastructure       0.99      0.82      0.90       221
       weather_related       0.98      0.94      0.96      1452
                floods       0.99      0.88      0.93       432
                 storm       0.98      0.89      0.94       480
                  fire       1.00      0.84      0.91        50
            earthquake       0.98      0.95      0.96       460
                  cold       0.99      0.82      0.89       103
         other_weather       1.00      0.80      0.89       286
         direct_report       0.97      0.85      0.91      1039
             micro avg       0.98      0.90      0.94     16808
             macro avg       0.96      0.81      0.88     16808
          weighted avg       0.98      0.90      0.93     16808
           samples avg       0.75      0.71      0.71     16808 
'''

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
