# pg_ml Machine learning module 
in development, don't stable

## folders
docs - documentation and slides

learn - learn notebooks

data - dumps of datasets
```
	-rw-rw-r-- 207344 ноя 16 20:36 adult.dmp.gz
	-rw-rw-r--  17687 ноя 16 20:37 astra.dmp.gz
	-rw-rw-r--  13578 ноя 16 22:43 boston.dmp.gz
	-rw-rw-r--  12903 ноя 16 20:34 titanic.dmp.gz
```

## Prediction

The prediction has so far been made  for:
 - binary classification
 - multi classification
 - regression

## Installation

- export PG_HOME=/usr/local/pgsql    //where is main postgres folder

- wget https://github.com/catboost/catboost/releases/download/v1.2.2/libcatboostmodel.so

- git clone https://github.com/akalend/pg_ml.git

- cd pg_ml

- export PG_CONFIG=$PG_HOME/bin/pg_config

- export LD_LIBRARY_PATH=$PG_HOME/lib

- USE_PGXS=1 make

- sudo su

- export PATH=$PATH:$PG_HOME/bin

- USE_PGXS=1 make install

- chown postgres model.cbm

- [optional] cp model.cbm $PG_HOME/data


## examples

Titanic prediction model from CatBoost is 'titanic.cbt'.
Stars object model from CatBoost is 'astra3.cbt'.

Information about model:
```sql
SELECT ml_info('titanic.cbt');  
```

Prediction model:
```sql

SELECT ml_predict('path/to/model/model.cbt', 'test_tablename', '{Array,of,feature_categorial}');

-- binary classification. titanic dataset https://www.kaggle.com/competitions/titanic/data
SELECT ml_predict('titanic.cbt', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');  

-- adult dataset https://archive.ics.uci.edu/dataset/2/adult https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
select ml_predict('adult.cbm','adult2', 'workclass,education,marital-status,occupation,relationship,race,sex,native_country}');

-- multi classification https://www.kaggle.com/code/satoru90/stellar-classification-dataset-sdss17/
SELECT ml_predict('astra3.cbt', 'astra');

--- regression   https://www.kaggle.com/c/boston-housing
SELECT ml_predict('boston.cbt', 'boston');

```

## Before compilation
You need add the libcatboostmodel.so to postgres library dir: /usr/local/pgsql/lib

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/pgsql/lib
```

More detail information You can see in the docs/pg_ml.pdf
