/*
 * ml.c
 *
 * Alexandre Kalendarev <akalend@mail.ru>
 * 
 *  SELECT ml_predict ('titanic.cbm', 'titanic','{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');
 *  SELECT ml_predict ('adult.cbm',   'adult2','{workclass,education,marital_status, occupation,relationship,race,sex,native_country}');
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>

#include "postgres.h"
#include "c_api.h"


#include "access/htup.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/varlena.h"
#include "utils/numeric.h"

//#include "common/jsonapi.h"
#include "utils/json.h"
#include "utils/jsonb.h"
#include "utils/jsonfuncs.h"



#define ML_VERSION "PgCatBoost 0.0.3"
#define FIELDLEN 64

typedef struct ArrayDatum {
    int count;
    Datum *elements; 
} ArrayDatum;


static double sigmoid(double x);
static const char* getModelParms(ModelCalcerHandle* modelHandle);
static char* getModelType(ModelCalcerHandle *modelHandle, const char* info);
static char*** getModelClasses(ModelCalcerHandle* modelHandle, const char* info);
static char** GetModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount);

static void CretatePredictTable(char* tablename, char* modelType);
static void DropPredictTable(char* tablename);
static void LoadModel(text  *filename, ModelCalcerHandle **model);

static bool checkInArray(char *name, char **features, int featureCount);
static bool checkInTextArray(char *name, ArrayDatum *featureArray);
static bool pstrcasecmp(char *s1, char *s2);
static void predict(ModelCalcerHandle  *modelHandle, char  *tabname, ArrayDatum *cat_fields, char* modelType, char*** modelClasses);


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(ml_version);
PG_FUNCTION_INFO_V1(ml_predict);
PG_FUNCTION_INFO_V1(ml_cat_predict);
PG_FUNCTION_INFO_V1(ml_test);
PG_FUNCTION_INFO_V1(ml_info);

static double
sigmoid(double x) {
    return 1. / (1. + exp(-x));
}


Datum
ml_version(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(ML_VERSION));
}


static void
DropPredictTable(char* tablename)
{
    StringInfoData buf;
    initStringInfo(&buf);

    appendStringInfo(&buf,
        "DROP TABLE IF EXISTS public.%s_predict;",
        tablename);
    
    SPI_execute(buf.data, false, 0);
    pfree(buf.data);
}

static void 
CretatePredictTable(char* tablename, char* modelType)
{
    StringInfoData buf;
    int res;

    initStringInfo(&buf);
    appendStringInfo(&buf,
        "DROP TABLE IF EXISTS public.%s_predict;",
        tablename);
    
    res = SPI_execute(buf.data, false, 0);
    if(res != SPI_OK_UTILITY )
    {
        elog(ERROR,"error Query: %s", buf.data);
    }


    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "CREATE TABLE IF NOT EXISTS public.%s_predict AS SELECT row_number() over() as row,* FROM %s;",
        tablename, tablename);
    
    res = SPI_execute(buf.data, false, 0);
    if(res != SPI_OK_UTILITY )
    {
        elog(ERROR,"error Query: %s", buf.data);
    }

    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "ALTER TABLE IF EXISTS %s_predict SET SCHEMA public;",
        tablename);

    res = SPI_execute(buf.data, false, 0);
    if(res != SPI_OK_UTILITY )
    {
        elog(ERROR,"error Query: %s", buf.data);
    }

    resetStringInfo(&buf);
    appendStringInfo(&buf,
        "ALTER TABLE IF EXISTS public.%s_predict ADD COLUMN predict FLOAT;",
        tablename);
    res = SPI_execute(buf.data, false, 0);
    if(res != SPI_OK_UTILITY )
    {
        elog(ERROR,"error Query: %s", buf.data);
    }

    if(strcmp(modelType, "\"RMSE\"") != 0) {
        resetStringInfo(&buf);
        appendStringInfo(&buf,
        "ALTER TABLE IF EXISTS public.%s_predict ADD COLUMN class TEXT;",
        tablename);
        res = SPI_execute(buf.data, false, 0);
        if(res != SPI_OK_UTILITY )
        {
            elog(ERROR,"error Query: %s", buf.data);
        }        
    }

    resetStringInfo(&buf);
    pfree(buf.data);
}


/*
* check filename and load CatBosot model
*/
static void 
LoadModel(text  *filename, ModelCalcerHandle** modelHandle)
{
    struct stat buf;
    char  *filename_str = text_to_cstring(filename);
    if (stat(filename_str, &buf) == -1)
    {
        int         err = errno;
        elog(ERROR, "file %s has error: %d:%s",filename_str, err, strerror(err));
    }

    *modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(*modelHandle, filename_str)) {
        elog(ERROR, "LoadFullModelFromFile error message: %s\n", GetErrorString());
    }
}

/*
*  get array names Model features
*/
static char** 
GetModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount)
{
    char** featureName = palloc(1024);

    bool rc = GetModelUsedFeaturesNames(modelHandle, &featureName, featureCount);

    if (!rc) {
        elog(ERROR,"get model feature name error");
    }
    return featureName;
}


static bool
checkInArray(char* name, char **features, int featureCount)
{
    int i;
    for(i=0; i < featureCount; i++)
    {
        if ( pstrcasecmp(features[i], name) ){
            // elog(NOTICE, "%d %s/%s %d",i, features[i],name, featureCount);
            return true;            
        }
            
    }
    return false;
}


static bool
checkInTextArray(char* name, ArrayDatum * featuresArr)
{
    int i;
    Datum* p;
    if (featuresArr == NULL){
        return false;
    }

    p = featuresArr->elements;

    for(i=0; i < featuresArr->count; i++)
    {
        char* fieldName = TextDatumGetCString(*p);

        if ( pstrcasecmp(fieldName, name) )
        {
            return true;
        }
        p ++;
    }
    return false;
}


/*
* case compare column name and model feature name
* and replace symbol '-' to '_'
* as the postgers can't use symbol '-' in column name
*/
static bool
pstrcasecmp(char  *s1, char  *s2)
{
    char *p1,*p2, pp1, pp2;
    p1=s1;
    p2=s2;

    while (*p1 && *p2)
    {
        if (isalpha(*p1))
            pp1 = tolower(*p1);
        else
            if (*p1 == '-')
                pp1 = '_';
            else
                pp1 = *p1;

        if (isalpha(*p2))
            pp2 = tolower(*p2);
        else
            if (*p2 == '-')
                pp2 = '_';
            else
                pp2 = *p2;

        if (pp1 == '_' && strcasecmp(p1+1,"id") == 0 && strcasecmp(p2,"ID") == 0)
            return true;

        if (pp2 == '_' && strcasecmp(p2+1,"id") == 0 && strcasecmp(p1,"ID") == 0)
            return true;

        if (pp1!= pp2)
        {
            return false;
        }
        
        p1 ++;
        p2 ++;
    }

    if (*p1 !=*p2)
        return false;

    return true;
}

static void
predict(ModelCalcerHandle* modelHandle, char* tabname, ArrayDatum* cat_fields, char *modelType, char ***modelClasses)
{
    StringInfoData buf;
    SPITupleTable *spi_tuptable;
    TupleDesc   spi_tupdesc;
    int i, j, res, rows;
    int cat_feature_counter = 0;
    int feature_counter = 0;
    int *iscategory;
    int model_float_feature_count;
    int model_cat_feature_count ;
    int model_dimension;
    size_t featureCount = 0;
    char **features;
    float *row_fvalues;
    int memsize;
    char *cat_value_buffer;
    char **row_cvalues;
    double *result_pa;
    double *result_exp;
    bool isMultiClass = false;
    int fieldCount = 0;

    if (strcmp(modelType, "\"MultiClass\"") == 0)
        isMultiClass = true;       


    initStringInfo(&buf);
    appendStringInfo(&buf, "SELECT * FROM public.%s_predict", tabname);

    res = SPI_exec(buf.data, 0);
    
    if (res < 1 || SPI_tuptable == NULL)
    {
        elog(ERROR, "Query %s error", buf.data);
    }
    rows = SPI_processed;

    spi_tuptable = SPI_tuptable;
    if (spi_tuptable == NULL)
        elog(ERROR,"tuptable is null");

    spi_tupdesc = spi_tuptable->tupdesc;

    iscategory = palloc( spi_tupdesc->natts * sizeof(int));
    model_float_feature_count = (int)GetFloatFeaturesCount(modelHandle);
    model_cat_feature_count = (int)GetCatFeaturesCount(modelHandle);
    model_dimension = (int)GetDimensionsCount(modelHandle);


    features = GetModelFeatures(modelHandle, &featureCount);

    fieldCount = isMultiClass ? spi_tupdesc->natts -1 : spi_tupdesc->natts;

    for(i=1; i < fieldCount; i++)
    {
        
        if(! checkInArray(spi_tupdesc->attrs[i].attname.data, features, featureCount))
        {
            iscategory[i] = -1;
            continue;
        } 
        
        if ( checkInTextArray(spi_tupdesc->attrs[i].attname.data, cat_fields) )
        {
            cat_feature_counter ++;
            iscategory[i] = 1;
        }
        else
        {
            feature_counter ++;
            iscategory[i] = 0;
        }
    }

     //  check model features
    if (feature_counter != model_float_feature_count)
    {
        DropPredictTable(tabname);
        elog(ERROR,
            "count of numeric features is not valid, must be %d is %d",
            model_float_feature_count, feature_counter
        );
        return;
    }

    if (cat_feature_counter != model_cat_feature_count)
    {
        DropPredictTable(tabname);
        elog(ERROR,
            "count of categocical features is not valid, must be %d is %d",
            model_cat_feature_count, cat_feature_counter
        );
        return;
    }

    row_fvalues = palloc( model_float_feature_count * sizeof(float));
    memsize = model_cat_feature_count * sizeof(char) * FIELDLEN;
    cat_value_buffer  = palloc(memsize);
    row_cvalues = palloc(model_cat_feature_count * sizeof(char*));

    result_pa  = (double*) palloc( sizeof(double) * model_dimension);
    result_exp = (double*) palloc( sizeof(double) * model_dimension);


    for (i = 0; i < rows; i++)
    {
        char *p = (char*)cat_value_buffer;
        HeapTuple   spi_tuple;
        feature_counter = 0;
        cat_feature_counter = 0;

        spi_tuple = spi_tuptable->vals[i];

        for(j=1; j < fieldCount; j++)
        {
            char *value;

            value = SPI_getvalue(spi_tuple, spi_tupdesc, j+1);

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


        if (!CalcModelPredictionSingle(modelHandle, 
                    row_fvalues, feature_counter,
                    (const char** )row_cvalues, cat_feature_counter,
                    result_pa, model_dimension)
            )
        {
            StringInfoData str;
            initStringInfo(&str);
            for(j = 0; j < feature_counter; j++)
            {
                appendStringInfo(&str, "%f,", row_fvalues[j]);
            }
            elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%d", GetErrorString(), i);
        }

        resetStringInfo(&buf);

        if (isMultiClass)
        {

            char  ***p;
            double max = 0., sm = 0.;
            int max_i = -1;

            for( j = 0; j < model_dimension; j ++)
            {
                result_exp[j] = exp(result_pa[j]);
                sm += result_exp[j];
            }
            for( j = 0; j < model_dimension; j ++)
            {
                result_exp[j] = result_exp[j] / sm;
                if (result_exp[j] > max){
                    max = result_exp[j];
                    max_i = j;
                }
            }

            p = modelClasses + max_i;

            appendStringInfo(&buf, "UPDATE public.%s_predict SET predict=%f,class='%s' WHERE row=%s;", tabname, max ,(char*)*p ,SPI_getvalue(spi_tuple, spi_tupdesc, 1));

        }
        else if (strcmp(modelType, "\"RMSE\"") == 0)
        {
            appendStringInfo(&buf, "UPDATE public.%s_predict SET predict=%f WHERE row=%s;", tabname, result_pa[0],SPI_getvalue(spi_tuple, spi_tupdesc, 1));
        }
        else
        {
            double probability = sigmoid(result_pa[0]);
            int n = 0;

            if (probability > 0.5)
            {
                n = 1;
            }


            appendStringInfo(&buf, "UPDATE public.%s_predict SET predict=%f, class='%s' WHERE row=%s;",
             tabname, probability,(char*)*( modelClasses + n),SPI_getvalue(spi_tuple, spi_tupdesc, 1));
        }

        SPI_execute(buf.data, false, 0);

    }       // dataset

    // pfree(result_pa); // ???
}


Datum
ml_cat_predict(PG_FUNCTION_ARGS)
{


    StringInfoData buf;
    ModelCalcerHandle* modelHandle;
    Datum      *key_datums;
    bool       *key_nulls;
    int         key_count;
    ArrayType  *key_array;
    ArrayDatum  cat_fields;
    const char*       model_info;
    char*** modelClasses;
    // char **featureName;
    // int featureCount;
    char* modelType;

    text *filename = PG_GETARG_TEXT_PP(0);
    char *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));


    // ------------- get categorical features  ----------------
    key_array = PG_GETARG_ARRAYTYPE_P(2);

    if (key_array == NULL) {
        elog(ERROR, "key is null");
    }

    Assert(ARR_ELEMTYPE(key_array) == TEXTOID);


    deconstruct_array(key_array,
                      TEXTOID, -1, false, TYPALIGN_INT,
                      &key_datums, &key_nulls, &key_count);

    cat_fields.elements = key_datums;
    cat_fields.count = key_count;

    LoadModel(filename, &modelHandle);

    SPI_connect();
    model_info = getModelParms(modelHandle);
    modelClasses= getModelClasses(modelHandle, model_info);
    modelType = getModelType(modelHandle, model_info);
    CretatePredictTable(tabname, modelType);

    predict(modelHandle, tabname, &cat_fields, modelType, modelClasses);


    if (SPI_finish() != SPI_OK_FINISH)
        elog(WARNING, "could not finish SPI");

    ModelCalcerDelete(modelHandle);

    initStringInfo(&buf);
    appendStringInfo(&buf, "public.%s_predict", tabname);
    

    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}



Datum
ml_predict(PG_FUNCTION_ARGS)
{
    StringInfoData buf;
    ModelCalcerHandle* modelHandle;
    const char* model_info;
    char*** modelClasses;
    // char **featureName;
    // size_t featureCount;

    text *filename = PG_GETARG_TEXT_PP(0);
    char *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* modelType;

    LoadModel(filename, &modelHandle);

    SPI_connect();
    model_info = getModelParms(modelHandle);
    modelType = getModelType(modelHandle, model_info);
    CretatePredictTable(tabname, modelType);
    modelClasses = getModelClasses(modelHandle, model_info);

    predict(modelHandle, tabname, NULL, modelType, modelClasses);


    // SPI_commit();
    if (SPI_finish() != SPI_OK_FINISH)
		elog(WARNING, "could not finish SPI");

    ModelCalcerDelete(modelHandle);

    initStringInfo(&buf);
    appendStringInfo(&buf, "public.%s_predict", tabname);
    

    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}


static char*
getModelType(ModelCalcerHandle* modelHandle, const char* info)
{
    StringInfoData buf;
    TupleDesc tupdesc;
    SPITupleTable *tuptable;
    int ret;

    initStringInfo(&buf);

    if( !info )
    {
        appendStringInfo(&buf, "NULL");
        return buf.data;
    }

    appendStringInfo(&buf, "SELECT '%s'::jsonb #> '{loss_function,type}';", info);

    tuptable = SPI_tuptable;

    ret = SPI_exec(buf.data, 0);
    tuptable = SPI_tuptable;

    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tupdesc = tuptable->tupdesc;

    resetStringInfo(&buf);
    appendStringInfo(&buf, "%s", SPI_getvalue(tuptable->vals[0], tupdesc, 1));

    return buf.data;
}


static const char* getModelParms(ModelCalcerHandle* modelHandle)
{
    const char* info = GetModelInfoValue(modelHandle, "params", 6); // strlen("parms")
    if( !info )
    {
        return NULL;
    }
    return info;
}


static char***
getModelClasses(ModelCalcerHandle* modelHandle, const char* info)
{
    char ***res = NULL;
    StringInfoData buf;
    Jsonb* j;
    SPITupleTable *tuptable;
    int ret;
    bool is_null = false;
    Datum classes;
    TupleDesc tupdesc;

    if( !info )
    {
        return NULL;
    }

    initStringInfo(&buf);
    appendStringInfo(&buf, "SELECT '%s'::jsonb #> '{data_processing_options,class_names}';", info);

    tuptable = SPI_tuptable;
    ret = SPI_exec(buf.data, 0);
    tuptable = SPI_tuptable;

    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tupdesc = tuptable->tupdesc;

    if (0 == strcmp(SPI_getvalue(tuptable->vals[0], tupdesc, 1), "[]"))
    {
        return NULL;
    }

    classes = SPI_getbinval(tuptable->vals[0], tupdesc, 1,&is_null);

    if (is_null){
        elog(WARNING, "result is NULL");
        return NULL;
    }

    j = DatumGetJsonbP(classes);
    
    if(JB_ROOT_IS_ARRAY(j))
    {
        JsonbIterator *it;
        JsonbIteratorToken type;
        JsonbValue  jb;
        char*** p;

        res = (char***) palloc( sizeof(char*) * (getJsonbLength((const JsonbContainer*) j,1) + 1));
        p = res;
        it = JsonbIteratorInit(&j->root);

        while ((type = JsonbIteratorNext(&it, &jb, false))
               != WJB_DONE)
        {
            if (WJB_ELEM == type){
                if(jb.type == jbvString)
                {
                  *p = (char**)pnstrdup(jb.val.string.val, jb.val.string.len);
                   p++;
                }
                if(jb.type == jbvNumeric)
                {
                    Numeric num;
                    num = jb.val.numeric;

                    *p = (char**)pstrdup(DatumGetCString(DirectFunctionCall1(numeric_out,
                                              NumericGetDatum(num))));
                    p++;
                }
            }
        }
        *p = NULL;

        return res;
    }

    return NULL;
}



Datum
ml_info(PG_FUNCTION_ARGS)
{
    text *filename = PG_GETARG_TEXT_PP(0);
    StringInfoData buf;
    size_t featureCount;
    int i;
    char * modelType;
    const char* model_info;
    char *p;
    char ** features;

    ModelCalcerHandle* modelHandle;
    LoadModel(filename, &modelHandle);

    initStringInfo(&buf);

    appendStringInfo(&buf, "dimension:%ld numeric features:%ld categorial features:%ld",
            GetDimensionsCount(modelHandle),
            GetFloatFeaturesCount(modelHandle),
            GetCatFeaturesCount(modelHandle));

    SPI_connect();
    model_info = getModelParms(modelHandle);
    modelType = getModelType(modelHandle, model_info);
    SPI_finish();

    appendStringInfo(&buf," modelType %s\nfieldName:", modelType);
 
    features = GetModelFeatures(modelHandle, &featureCount);
    for(i=0; i < featureCount; i++)
    {
        appendStringInfo(&buf,"%s,", features[i]);
    }
    p = buf.data;
    *(p + buf.len -1 ) = ' ';

    ModelCalcerDelete(modelHandle);


    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}


Datum
ml_test(PG_FUNCTION_ARGS)
{

    PG_RETURN_TEXT_P(cstring_to_text("xx"));
}
