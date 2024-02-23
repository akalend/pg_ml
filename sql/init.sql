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
