# pg_ml Machine learning module

## folders
docs - documentation and slides

learn - learn notebooks

## Prediction

The prediction has so far been made only for binary classification.

## examples

Titanic prediction model from CatBoost is 'titanic.cbt'.

Information about model:
```sql
SELECT ml_info('titanic.cbt');  
```

Prediction model:
```sql
SELECT ml_predict('titanic.cbt', 'titanic');  
```

## Before compilation
You need add the libcatboostmodel.so to postgres library dir: /usr/local/pgsql/lib

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/pgsql/lib
```



