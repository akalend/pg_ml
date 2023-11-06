# pg_ml Machine learning module

## folders
docs - documentation and slides

learn - learn notebooks

## Prediction

The prediction has so far been made only for binary classification.

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

-- binary classification
SELECT ml_predict('titanic.cbt', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');  

-- multi classification
SELECT ml_predict('astra2.cbt', 'astra_test');  
```

## Before compilation
You need add the libcatboostmodel.so to postgres library dir: /usr/local/pgsql/lib

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/pgsql/lib
```



