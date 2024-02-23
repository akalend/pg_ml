CREATE EXTENSION ml;

SELECT ml_version();

SHOW ml.model_path;

SELECT ml_info('astra3.cbm');
SELECT ml_info('titanic.cbm');
SELECT ml_info('adult.cbm');


