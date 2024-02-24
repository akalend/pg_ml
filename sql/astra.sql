SELECT ml_predict_table('astra3.cbm', 'astra');

SELECT * FROM astra_predict LIMIT 3;

SELECT * FROM ml_predict('astra3.cbm', 'astra') LIMIT 3;

SELECT index as obj_id,class FROM ml_predict('astra3.cbm', 'astra', '{}', 'spec_obj_id') LIMIT 3;

SELECT *  FROM ml_predict_query('astra3.cbm', 'select * FROM astra LIMIT 3');

SELECT a.*,class FROM ml_predict('astra3.cbm', 'astra', '{}', 'spec_obj_id') as  p  , astra a WHERE p.index::text=a.spec_obj_id::text LIMIT 3 ;