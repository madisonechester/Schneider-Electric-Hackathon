# SE-Europe-Data_Challenge
## Description
This project is dedicated to building a predictive model for determining which of the following countries - Hungary (HU), Italy (IT), Poland (PO), Spain (SP), Germany (DE), Denmark (DK), Sweden (SE), and the Netherlands (NE) - will have the largest green energy surplus in the upcoming hour. This initiative was undertaken as a part of the Schneider Electric Hackathon.

## Installation
To set up the project environment, ensure that you have Python installed on your system. Then, run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage
### Data Ingestion
To run this project first you should get the data from the ENTSO-E API.
 ```
 python data_ingestion.py
 ``` 

 This will download a lot of files of load and generation from different sources. The only generation sources that should be kept are those that come from green energy.
 ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"].

 Furthermore, to replicate our results, you should drop the data coming from the UK, that we discarded due to it having too many missing values.


 ### Data Cleaning
 To clean the data and resample everything to 1H interval and get the columns needed for training run:

 ```
 python data_processing.py
 ```

This will generate a final_data.csv

### Model Training
To train the model run:
```
python model_training.py
```

The model is a LSTM neural network which predicts the country that will have the biggest surplus.

We also tried training a different model for each country and predict the surplus using LSTM and different features. This did not work as well. The notebook LSTM_predicting_surplus.ipynb contains our exploration of this method.

Another approach we took was using ARIMA and SARIMA, which we also could not make work. (testing ARIMA_SARIMA.ipynb)
### Predictions

To get the predictions run:
```
model_prediction.py
```

Our final F1 score was: F1:  0.8090072352366647

We chose a Baseline that was taking the last value as prediction. The F1 score of that is:

F1:  0.9577748468869401

In conclusion, our model does not beat the baseline so it is not useful.
## Authors and Acknowledgements

Thanks to the NUWE and Scheider Electric teams for hosting this event. We had a lot of fun even though we could not make the model work :)

- Adriana Álvaro ([@adrianaalvaro](https://github.com/adrianaalvaro))
- Madison Chester ([@madisonechester](https://github.com/madisonechester))
- Arturo Fredes ([@arturofredes](https://github.com/arturofredes))
