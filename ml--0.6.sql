
CREATE OR REPLACE FUNCTION ml_version()
RETURNS text
AS 'MODULE_PATHNAME','ml_version'
LANGUAGE C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_predict_table(
    model text,
    tablename text
) RETURNS text
AS 'MODULE_PATHNAME','ml_predict'
LANGUAGE C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_predict_table(
    model text,
    tablename text,
    categoirial_futures text[]
) RETURNS text
AS 'MODULE_PATHNAME','ml_cat_predict'
LANGUAGE C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_predict_internal(
    model text,
    tablename text,
    categoirial_futures text[] default '{}', 
    join_field text DEFAULT 'row',
    isQuery bool  DEFAULT FALSE,
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'MODULE_PATHNAME','ml_predict_dataset_inner'
LANGUAGE  C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_predict(
    model text,
    tablename text,
    categoirial_futures text[] default '{}', 
    join_field text DEFAULT 'row',
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'SELECT ml_predict_internal($1,$2,$3,$4, FALSE);'
LANGUAGE SQL VOLATILE STRICT;


CREATE OR REPLACE FUNCTION ml_predict_query(
    model text,
    query text,
    categoirial_futures text[] default '{}', 
    join_field text DEFAULT 'row',
    OUT index text,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'SELECT ml_predict_internal($1,$2,$3,$4, TRUE);'
LANGUAGE SQL VOLATILE STRICT;


CREATE OR REPLACE FUNCTION ml_info(model text )
RETURNS text
AS 'MODULE_PATHNAME','ml_info'
LANGUAGE C STRICT PARALLEL RESTRICTED;


CREATE OR REPLACE FUNCTION ml_json_info(
    model text,
    OUT feature text,
    OUT type text
) RETURNS setof RECORD
AS 'MODULE_PATHNAME','ml_json_parms_info'
LANGUAGE C STRICT PARALLEL RESTRICTED;




CREATE OR REPLACE FUNCTION ml_test(model text )
RETURNS RECORD
AS 'MODULE_PATHNAME','ml_test'
LANGUAGE C STRICT IMMUTABLE;
