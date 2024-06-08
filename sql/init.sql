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

CREATE TABLE public.rating_txt(
    id double precision,
    synopsis text,
    rating_mpaa text,
    genre text,
    director text,
    writer text,
    theater_date text,
    dvd_date text,
    box_office double precision,
    runtime double precision,
    studio text,
    dvd_date_int bigint,
    theater_date_int bigint,
    review text,
    rating double precision,
    fresh text,
    critic text,
    top_critic bigint,
    publisher text,
    date text,
    date_int double precision,
    rating_10 double precision
);


COPY rating_txt FROM '/tmp/model/rating_txt.csv' HEADER CSV; 
SELECT rating_10,critic,box_office FROM rating_txt LIMIT 4;

CREATE TABLE public.boston (
    id bigint,
    crim double precision,
    zn double precision,
    indus double precision,
    chas bigint,
    nox double precision,
    rm double precision,
    age double precision,
    dis double precision,
    rad bigint,
    tax bigint,
    ptratio double precision,
    black double precision,
    lstat double precision,
    medv double precision
);
COPY boston FROM '/tmp/model/boston.csv' HEADER CSV; 
SELECT * FROM boston LIMIT 3;

