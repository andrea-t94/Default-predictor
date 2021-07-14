# Default-predictor

## Abstract
Deep Learning shallow network for default prediction of users. The data preprocessing has been bone via classical ML techinques, such as ordinal encoding (for categorical value) and missing value management with mean value or zero, depending on the feature.

## Dataset
Sample Dataset for this particulare use case. Look at ```feature_config.py``` to see all the features involved, as well as configuration explaining how to deal with missing data.

## Requirements
- python == 3.8.5
- tensorflow==2.2.0 for Model build and training
- scikit-learn==0.24.2 for advanced data preprocessing techinques for NLP (e.g. tfidf)
- pandas==1.2.4 for data ingestion and management
- gunicorn==20.1.0 for server application \
For the other requirements check out requirements.txt

## Project file structure

```
/dataset.csv
/Dockerfile
/Dockerrrun.aws.json #helper file to run on AWS Beanstalk
/train.py
/inference.py
/app.py
/wsgi.py #web server application
/models #in which stored model trained
/artifacts #in which stored text processor trained
/featureAnalysis #notebook with some feature distribution analysis
/...
```

## Installation
For Docker install check out [official documentation](https://docs.docker.com/get-docker/)
- clone repository ```
$ git clone https://github.com/andrea-t94/Default-predictor.git ```
- build docker image ```
$ docker build -t default-predictor -f Dockerfile . ```\
- host gunicorn app ```
$ docker run -p 5000:5000 default-predictor ```\
\
Docker build will:
- install all the dependencies
- run a training for deploying the first model (NB. the train will deal with the first model and processor deployment)
- run inference on the missing default of the dataset and output it in csv format\

## How to run Docker
Once built the docker image:
- ```$ docker run default-predictor python3 train.py ``` will handle the training and store the new text processor in artifacts directory and the model in model directory (not necessary, since in the build a model has already been trained)
- ```$ docker run default-predictor python3 inference.py  ``` will return the output of the inference of the records missing of dafault value in the dataset
- ```$ docker run -p 5000:5000 default-predictor ``` will host the API on gunicorn application server, callable at http://localhost:5000/

## How to query on local/AWS hosted API
- Gunicorn Deployment (manual) ``` $ curl -X POST "http://0.0.0.0:5000/api/detection" -H "Content-Type: application/json" --data '{JSON DATA}' ```
- AWS deployed ``` $ curl -X POST "http://defaultpredictorat-env-1.eba-qpsnfdxf.eu-west-2.elasticbeanstalk.com/api/detection" -H "Content-Type: application/json" --data  '{JSON DATA}' ```
## NB: Please refer to ``` test_curl.sh ``` to an example of how to query both API, from command line

## Input Data
- The API can handle missing values, both in training and inference except for the categorical features  "merchant_category","merchant_group", "has_paid", "name_in_email"
- JSON DATA structure sample ``` '{"account_amount_added_12_24m":"0",
 "account_days_in_dc_12_24m":"0",
 "account_days_in_rem_12_24m":"0",
 "account_days_in_term_12_24m":"0",
 "account_incoming_debt_vs_paid_0_24m":"0",
 "account_status":"1",
 "account_worst_status_0_3m":1,
 "account_worst_status_12_24m":"",
 "account_worst_status_3_6m":"1",
 "account_worst_status_6_12m":"",
 "age":"20",
 "avg_payment_span_0_12m":"12.69230769",
 "avg_payment_span_0_3m":"8.333333333",
 "merchant_category":"Dietary supplements",
 "merchant_group":"Health & Beauty",
 "has_paid":true,
 "max_paid_inv_0_12m":"31638",
 "max_paid_inv_0_24m":"31638",
 "name_in_email":"no_match",
 "num_active_div_by_paid_inv_0_12m":"0.153846154",
 "num_active_inv":"2",
 "num_arch_dc_0_12m":"0",
 "num_arch_dc_12_24m":"0",
 "num_arch_ok_0_12m":"13",
 "num_arch_ok_12_24m":"14",
 "num_arch_rem_0_12m":"0",
 "num_arch_written_off_0_12m":"0",
 "num_arch_written_off_12_24m":"0",
 "num_unpaid_bills":"2",
 "status_last_archived_0_24m":"1",
 "status_2nd_last_archived_0_24m":"1",
 "status_3rd_last_archived_0_24m":"1",
 "status_max_archived_0_6_months":"1",
 "status_max_archived_0_12_months":"1",
 "status_max_archived_0_24_months":"1",
 "recovery_debt":"0",
 "sum_capital_paid_account_0_12m":"0",
 "sum_capital_paid_account_12_24m":"0",
 "sum_paid_inv_0_12m":"178839",
 "time_hours":"9.653333333",
 "worst_status_active_inv":"1"
 }' ```

## Output Data
the output, in JSON format:
``` {"message":
  {"ESTIMATE":"1.0", #probability of default, string format
  "MISSING_DATA":true,
  "MISSING_DATA_LIST":"['account_worst_status_12_24m', 'account_worst_status_6_12m']". #list of missing features properly handled
 } ```
 
