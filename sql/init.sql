CREATE TABLE astra (
    alpha double precision,
    delta double precision,
    u double precision,
    g double precision,
    r double precision,
    i double precision,
    z double precision,
    run_id bigint,
    cam_col bigint,
    field_id bigint,
    spec_obj_id double precision,
    redshift double precision,
    plate bigint,
    mjd bigint,
    fiber_id bigint
);

COPY astra FROM '/tmp/model/astra.csv' HEADER CSV; 

SELECT * FROM astra LIMIT 2;

CREATE TABLE public.titanic (
    id integer,
    passenger_id integer,
    pclass integer,
    name text,
    sex text,
    age double precision,
    sibsp integer,
    parch integer,
    ticket text,
    fare double precision,
    cabin text,
    embarked character(1),
    res boolean
);

COPY titanic FROM '/tmp/model/titanic.csv' HEADER CSV; 
SELECT * FROM titanic LIMIT 2;
