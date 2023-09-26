/*
 * ml.c
 *
 * Alexandre Kalendarev <akalend@mail.ru>
 *
 *
 */

#include "postgres.h"
#include "c_api.h"

#include <limits.h>
#include <math.h>

#include "access/htup.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/varlena.h"

#define ML_VERSION "PgCatBoost 0.1"
#define FIELDLEN 64
PG_MODULE_MAGIC;

/*
 * This is the trigger that protects us from orphaned large objects
 */
PG_FUNCTION_INFO_V1(ml_version);
PG_FUNCTION_INFO_V1(ml_predict);
PG_FUNCTION_INFO_V1(ml_test);
PG_FUNCTION_INFO_V1(ml_info);

double
sigmoid(double x);

double
sigmoid(double x) {
    return 1. / (1. + exp(-x));
}


Datum
ml_version(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(ML_VERSION));
}

Datum
ml_predict(PG_FUNCTION_ARGS)
{
    StringInfoData buf;
    HeapTuple   tuple;
    SPITupleTable *spi_tuptable;
    TupleDesc   spi_tupdesc;
    int i,j;

    text *filename = PG_GETARG_TEXT_PP(0);

    ModelCalcerHandle* modelHandle;
    modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(modelHandle, text_to_cstring(filename)  )) {
        elog(ERROR, "LoadFullModelFromFile error message: %s\n", GetErrorString());
    }
    int model_float_feature_count = (int)GetFloatFeaturesCount(modelHandle);
    int model_cat_feature_count = (int)GetCatFeaturesCount(modelHandle);

    char *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));


    initStringInfo(&buf);

    SPI_connect();

    appendStringInfo(&buf,
        "DROP TABLE IF EXISTS public.%s_predict;",
        tabname);
    
    int res = SPI_execute(buf.data, false, 0);
    // elog(WARNING,"%s res=%d", buf.data, res);

    appendStringInfo(&buf,
        "CREATE TABLE IF NOT EXISTS public.%s_predict AS SELECT row_number() over() as row,* FROM %s;",
        tabname, tabname);
    
    res = SPI_execute(buf.data, false, 0);
    // elog(WARNING,"%s res=%d", buf.data, res);
    
    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "ALTER TABLE IF EXISTS %s_predict SET SCHEMA public;",
        tabname);
    
    res = SPI_execute(buf.data, false, 0);
    // elog(WARNING,"%s res=%d", buf.data, res);
    resetStringInfo(&buf);

    appendStringInfo(&buf,
        "ALTER TABLE IF EXISTS public.%s_predict ADD COLUMN predict FLOAT;",
        tabname);
    
    res = SPI_execute(buf.data, false, 0);
    
    // elog(WARNING,"%s res=%d", buf.data, res);
    resetStringInfo(&buf);



    appendStringInfo(&buf, "SELECT * FROM public.%s_predict", tabname);

    // elog(WARNING,"%s", buf.data);
    res = SPI_exec(buf.data, 0);


    spi_tuptable = SPI_tuptable;

	if (res < 1 || spi_tuptable == NULL)
    {
    	elog(ERROR, "Query error");
    	PG_RETURN_NULL();
    }

    spi_tupdesc = spi_tuptable->tupdesc;

    int *iscategory = palloc( spi_tupdesc->natts * sizeof(int));
    
    int cat_feature_counter = 0;
    int feature_counter = 0;
    bool warning_flag = false;



    for(i=1; i < spi_tupdesc->natts-1; i++)
    {
        switch (spi_tupdesc->attrs[i].atttypid)
        {
        case 18:        // char
        case 25:        // varchar            
        case 1043:      // text
            cat_feature_counter ++;
            iscategory[i] = 1;
            break;
        case 20:        // int8
        case 21:        // int2
        case 23:        // int4
        case 700:       // float4
        case 701:       // float8
            feature_counter ++;
            iscategory[i] = 0;
            break;
        default:
            iscategory[i] = -1;
            warning_flag = true;
        }
        // elog(WARNING, "%s %d", spi_tupdesc->attrs[i].attname.data, iscategory[i]);
    }
    
    if (warning_flag)
    {
        for(i=0; i < spi_tupdesc->natts; i++)
        {
            if (iscategory[i] == -1)
                elog(WARNING, "check type of field %s", spi_tupdesc->attrs[i].attname.data);
        }

    }

    if (feature_counter != model_float_feature_count)
    {
        elog(ERROR,
            "count of numeric features is not valid, must be %d is %d",
            model_float_feature_count, feature_counter
        );
        PG_RETURN_NULL();
    }

    if (cat_feature_counter != model_cat_feature_count)
    {
        elog(ERROR,
            "count of categocical features is not valid, must be %d is %d",
            model_cat_feature_count, cat_feature_counter
        );
        PG_RETURN_NULL();
    }

    float *row_fvalues = palloc( model_float_feature_count * sizeof(float));
    int memsize = model_cat_feature_count * sizeof(char) * FIELDLEN;
    char *cat_value_buffer  = palloc(memsize);
    char **row_cvalues = palloc(model_cat_feature_count * sizeof(char*));


    double result_a[1];
    double *result_pa = result_a;
    int rows = SPI_processed;

    for (i = 0; i < rows; i++)
    {
    	char *p = (char*)cat_value_buffer;
        HeapTuple   spi_tuple;
        feature_counter = 0;
        cat_feature_counter = 0;
        spi_tuple = spi_tuptable->vals[i];

        resetStringInfo(&buf);
        char *row = SPI_getvalue(spi_tuple, spi_tupdesc, 1);

        for(j=1; j < spi_tupdesc->natts-1; j++)
        {
            char *value = SPI_getvalue(spi_tuple, spi_tupdesc, j+1);
            char *name = spi_tupdesc->attrs[j].attname.data; 

            // elog(WARNING,"%s value[%d]='%s'",name,j,value);
            
            if( iscategory[j] == 0)
            {
                int res  = sscanf(value, "%f", &row_fvalues[feature_counter]);
                if(res < 1)
                {
                    elog(WARNING,"error input j/cnt=%d/%d\n", j, feature_counter);
                }
                feature_counter++;
            }

            if( iscategory[j] == 1)
            {
                row_cvalues[cat_feature_counter] = strcpy(p, value);
                cat_feature_counter++;
                p += strlen(value);
                *p = '\0';
                p++;
            }

        }   // column

        if (!CalcModelPrediction(modelHandle, 1,
                    (const float **) &row_fvalues, feature_counter,
                    (const char***)&row_cvalues, cat_feature_counter,
                    result_pa, 1)
            )
        {
            elog( ERROR, "CalcModelPrediction error message: %s row=%d", GetErrorString(), i);
        }

        double probability = sigmoid(result_a[0]);
        
        appendStringInfo(&buf, "UPDATE %s_predict SET predict=%f WHERE row=%s;", tabname, probability,row);
    	SPI_execute(buf.data, false, 0);
    	// elog(WARNING,"%s", buf.data);

    }       // dataset

    // SPI_commit();
    if (SPI_finish() != SPI_OK_FINISH)
		elog(WARNING, "could not finish SPI");

    ModelCalcerDelete(modelHandle);

    resetStringInfo(&buf);
    appendStringInfo(&buf, "public.%s_predict", tabname);
    

    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}



Datum
ml_info(PG_FUNCTION_ARGS)
{
    text *filename = PG_GETARG_TEXT_PP(0);
    StringInfoData buf;

    ModelCalcerHandle* modelHandle;
    modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(modelHandle, text_to_cstring(filename)  )) {
        elog(ERROR, "LoadFullModelFromFile error message: %s\n", GetErrorString());
    }

    initStringInfo(&buf);

    appendStringInfo(&buf, "dimension:%ld numeric features:%ld cagorial features:%ld",
            GetDimensionsCount(modelHandle),
            GetFloatFeaturesCount(modelHandle),
            GetCatFeaturesCount(modelHandle));


    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}