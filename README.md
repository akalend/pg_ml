# pg_ml Machine learning module 
in development, don't stable

## folders
docs - documentation and slides

learn - train model notebooks

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
SELECT ml_info('amazon.cbm');
                                                        ml_info                                                         
------------------------------------------------------------------------------------------------------------------------
 dimension:1 numeric features:0 categorial features:9 modelType "Logloss"                                              +
 fieldName:RESOURCE,MGR_ID,ROLE_ROLLUP_1,ROLE_ROLLUP_2,ROLE_DEPTNAME,ROLE_TITLE,ROLE_FAMILY_DESC,ROLE_FAMILY,ROLE_CODE 
(1 row)

SELECT ml_info('titanic.cbm');
                                      ml_info                                      
-----------------------------------------------------------------------------------
 dimension:1 numeric features:2 categorial features:9 modelType "Logloss"         +
 fieldName:PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked 
(1 row)

--- If You save train model as json format
SELECT ml_json_info('astra.json');
NOTICE:  rows=12
NOTICE:  rows=0
                               ml_json_info                               
--------------------------------------------------------------------------
 float feature:alpha,delta,u,g,r,i,z,cam_col,redshift,plate,MJD,fiber_ID,+
 categorial feature:
(1 row)

```

Prediction model:
```sql

SELECT ml_predict('path/to/model/model.cbt', 'test_tablename', '{Array,of,feature_categorial}');

-- binary classification. titanic dataset https://www.kaggle.com/competitions/titanic/data
SELECT ml_predict('titanic.cbt', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');  

-- adult dataset https://archive.ics.uci.edu/dataset/2/adult https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
SELECT ml_predict('adult.cbm','adult2', 'workclass,education,marital-status,occupation,relationship,race,sex,native_country}');

-- multi classification https://www.kaggle.com/code/satoru90/stellar-classification-dataset-sdss17/
--- create tabe: public.astra3_predict, as table 'astra' with new added columns: predict and class
 SELECT ml_predict_table('astra3.cbm', 'astra');
   ml_predict_table    
-----------------------
 public.astra_predict
(1 row)

SELECT * FROM astra_predict;
row |       alpha       |       delta        |    u     |    g     |    r     |    i     |    z     | run_id | cam_col | field_id |      spec_obj_id       |   redshift    | plate |  mjd  | fiber_id | predict  | class  
------------------------+--------------------+----------+----------+----------+----------+----------+--------+---------+----------+------------------------+---------------+-------+-------+----------+----------+--------
   1 |  16.9568897845004 |   3.64613008870454 | 23.33542 | 21.95143 | 20.48149 |   19.603 | 19.13094 |   7712 |       6 |      442 |  4.855016555329904e+18 |     0.5062369 |  4312 | 55511 |      495 |  0.98686 | GALAXY
   2 |  240.063240247767 |   6.13413059813973 | 17.86033 | 16.79228 | 16.43001 | 16.30923 | 16.25873 |   3894 |       1 |      243 | 2.4489280322708705e+18 |  0.0003448142 |  2175 | 54612 |      348 | 0.990419 | STAR


--- regression   https://www.kaggle.com/c/boston-housing
SELECT ml_predict_table('boston.cbt', 'boston');

--- the dataset of the prediction
SELECT * from  ml_cat_predict('boston.cbm', 'boston2') LIMIT 3;
 row_num |      predict       | class 
---------+--------------------+-------
       0 |  24.99982028068538 | 
       1 | 20.664358727562394 | 
       2 |  33.67737911788664 | 
(3 rows)

--- You can join dataset with other tables
WITH  predict  AS (  SELECT * from ml_cat_predict('astra3.cbm', 'astra'))
    SELECT * FROM astra a 
       LEFT JOIN predict p ON (p.row_num = a.row)   ;
 row |       alpha       |       delta        |    u     |    g     |    r     |    i     |    z     | run_id | cam_col | field_id |      spec_obj_id       |   redshift    | plate |  mjd  | fiber_id | row_num |      predict       | class  
-----+-------------------+--------------------+----------+----------+----------+----------+----------+--------+---------+----------+------------------------+---------------+-------+-------+----------+---------+--------------------+--------
   0 |  16.9568897845004 |   3.64613008870454 | 23.33542 | 21.95143 | 20.48149 |   19.603 | 19.13094 |   7712 |       6 |      442 |  4.855016555329904e+18 |     0.5062369 |  4312 | 55511 |      495 |       0 | 0.9868595777513302 | GALAXY
   1 |  240.063240247767 |   6.13413059813973 | 17.86033 | 16.79228 | 16.43001 | 16.30923 | 16.25873 |   3894 |       1 |      243 | 2.4489280322708705e+18 |  0.0003448142 |  2175 | 54612 |      348 |       1 | 0.9904188657285139 | STAR
   2 |   30.887222067625 |   1.18870964120799 | 18.18911 | 16.89469 | 16.42161 | 16.24627 | 16.18549 |   7717 |       1 |      536 |  8.255357438959835e+18 |  4.085216e-06 |  7332 | 56683 |      943 |       2 | 0.9975875623929414 | STAR
   3 |  247.594400505002 |   10.8877797153666 | 24.99961 | 21.71203 | 21.47148 | 21.30532 | 21.29109 |   5323 |       1 |      134 |  4.577998722756271e+18 | -0.0002914838 |  4066 | 55444 |      326 |       3 | 0.9976669380943318 | STAR
   4 |  18.8964507920807 |  -5.26133022886992 | 23.76648 | 21.79737 | 20.69543 | 20.23403 | 19.97464 |   7881 |       3 |      148 |   8.91047176642785e+18 | -0.0001361561 |  7914 | 57331 |      363 |       4 | 0.9960439244920889 | STAR
   5 |  182.713733094955 |   51.3758050594777 | 22.44608 | 21.68444 | 20.24292 | 19.41423 | 19.08227 |   2830 |       1 |      411 |  7.516725588574623e+18 |     0.5026683 |  6676 | 56389 |      792 |       5 | 0.9843734017027631 | GALAXY



```

## Before compilation
You need add the libcatboostmodel.so to postgres library dir: /usr/local/pgsql/lib

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/pgsql/lib
```

More detail information You can see in the docs/pg_ml.pdf
