
CREATE FUNCTION ml_version()
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

CREATE OR REPLACE FUNCTION ml_predict(
    model text,
    tablename text,
    categoirial_futures text[] default '{}', 
    OUT row_num bigint,
    OUT predict float,
    OUT class text
) RETURNS setof RECORD
AS 'MODULE_PATHNAME','ml_predict_dataset'
LANGUAGE C STRICT PARALLEL RESTRICTED;

CREATE FUNCTION ml_info(model text )
RETURNS text
AS 'MODULE_PATHNAME','ml_info'
LANGUAGE C STRICT PARALLEL RESTRICTED;

CREATE FUNCTION ml_json_feature_info(model text )
RETURNS text
AS 'MODULE_PATHNAME','ml_json_parms_info'
LANGUAGE C STRICT PARALLEL RESTRICTED;


CREATE FUNCTION ml_test(model text[] )
RETURNS text
AS 'MODULE_PATHNAME','ml_test'
LANGUAGE C STRICT IMMUTABLE;

CREATE OR REPLACE FUNCTION ml_json_info(model text )
RETURNS text
AS 'MODULE_PATHNAME','ml_json_parms_info'
LANGUAGE C STRICT PARALLEL RESTRICTED;
