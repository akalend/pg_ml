SELECT ml_predict_table('titanic.cbm', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');

SELECT passenger_id ,name,predict,class,res as result FROM titanic_predict limit 3;

SELECT index as name,class FROM ml_predict('titanic.cbm', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }', 'name') LIMIT 3;

SELECT index name,predict,class survived
  FROM ml_predict_query('titanic.cbm', 'select * FROM titanic LIMIT 3', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }', 'name');
SELECT * FROM ml_predict('titanic.cbm', 'titanic', '{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }') LIMIT 3;
