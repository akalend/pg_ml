/*
 * ml.c
 *
 * Alexandre Kalendarev <akalend@mail.ru>
 *
 *
 *  test this code:
 *  SELECT * FROM  ml_predict ('titanic.cbm', 'titanic','{name,passenger_id,pclass,sex,sibsp,parch,ticket,cabin,embarked }');
 *  SELECT * FROM ml_predict ('adult.cbm',   'adult2','{workclass,education,marital_status, occupation,relationship,race,sex,native_country}');
 *  select  json_array_elements((j #> '{features_info,float_features}')::json) #> '{feature_id}'     from parms  WHERE name='astra_all';
     SELECT name, j #> '{data_processing_options,class_names}' classes, j #>'{loss_function,type}' as loss  FROM parms ;

 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>

#include "postgres.h"
#include "c_api.h"
#include "funcapi.h"


#include "access/htup.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/varlena.h"
#include "utils/numeric.h"

#include "utils/json.h"
#include "utils/jsonb.h"
#include "utils/jsonfuncs.h"


#define ML_VERSION "PgCatBoost 0.5.0"
#define FIELDLEN    64
#define PAGESIZE    8192
#define QNaN        0x7fffffff
#define MAXDIGIT    12

#define ModelGetFieldName(i)  model->spi_tupdesc->attrs[i].attname.data

typedef struct ArrayDatum {
    int count;
    Datum *elements; 
} ArrayDatum;

typedef struct MLmodelData
{
    ModelCalcerHandle  *modelHandle;
    SPITupleTable      *spi_tuptable;
    TupleDesc           spi_tupdesc;
    char***             modelClasses;
    int8               *iscategory;
    float              *row_fvalues;
    char*              *row_cvalues;
    char*               cat_value_buffer;
    double             *result_pa;
    double             *result_exp;
    char*               keyField;
    char*               modelType;
    int64               current;
    int                 cat_count;
    int                 num_count;
    int                 attCount;
    int                 dimension;
    ArrayDatum          cat_fields;
} MLmodelData;

#define MLmodel MLmodelData*


static MemoryContext ml_context;
static char* model_path = "";

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
static void predict(ModelCalcerHandle  *modelHandle, char  *tabname,
                    ArrayDatum *cat_fields, char* modelType,
                    char*** modelClasses);
static Datum PredictGetDatum(char* id, int64 row_no, float8 predict, char* className,
                    TupleDesc tupleDescriptor);
static bool check_model_path(char **newval, void **extra, GucSource source);


void _PG_init(void);


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(ml_version);
PG_FUNCTION_INFO_V1(ml_predict);
PG_FUNCTION_INFO_V1(ml_cat_predict);
PG_FUNCTION_INFO_V1(ml_test);
PG_FUNCTION_INFO_V1(ml_info);
PG_FUNCTION_INFO_V1(ml_json_parms_info);

PG_FUNCTION_INFO_V1(ml_predict_dataset_inner);


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
        "CREATE TABLE IF NOT EXISTS public.%s_predict "
        "AS SELECT row_number() over() as row,* FROM %s;",
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
    StringInfoData sbuf;
    char slash[2] = "/\0";

    const char  *filename_str = text_to_cstring(filename);

    initStringInfo(&sbuf);
    if (strstr(filename_str, slash) == NULL)
    {
        int len = strlen( model_path);
        if (model_path[len-1] == '/')
            appendStringInfo(&sbuf, "%s%s", model_path, filename_str);
        else
            appendStringInfo(&sbuf, "%s/%s", model_path, filename_str);

        filename_str = sbuf.data;
    }


    if (stat(filename_str, &buf) == -1)
    {
        int         err = errno;
        elog(ERROR, "file %s has error: %d:%s",filename_str, err, strerror(err));
    }

    *modelHandle = ModelCalcerCreate();
    if (!LoadFullModelFromFile(*modelHandle, filename_str)) {
        elog(ERROR, "LoadFullModelFromFile error message: %s\n", GetErrorString());
    }
    resetStringInfo(&sbuf);
    pfree(sbuf.data);
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
    if (featuresArr == NULL || featuresArr->count == 0){
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
predict(ModelCalcerHandle* modelHandle, char* tabname,
        ArrayDatum* cat_fields, char *modelType, char ***modelClasses)
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
        
        if(! checkInArray(spi_tupdesc->attrs[i].attname.data,
                          features, featureCount))
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

    row_fvalues = palloc0( model_float_feature_count * sizeof(float));
    memsize = model_cat_feature_count * sizeof(char) * FIELDLEN;
    cat_value_buffer  = palloc0(memsize);
    row_cvalues = palloc0(model_cat_feature_count * sizeof(char*));

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
                if (value == 0 )
                {
                    row_fvalues[feature_counter] = QNaN;
                }
                else
                {
                    int res;
                    res  = sscanf(value, "%f", &row_fvalues[feature_counter]);
                    if(res < 1)
                    {
                        elog(WARNING,"error input j/cnt=%d/%d\n", j, feature_counter);
                    }
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
            elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%d",
                    GetErrorString(), i);
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

            appendStringInfo(&buf, "UPDATE public.%s_predict "
                                   "SET predict=%f,class='%s' "
                                   "WHERE row=%s;"
                                   , tabname, max ,(char*)*p ,
                                   SPI_getvalue(spi_tuple, spi_tupdesc, 1));

        }
        else if (strcmp(modelType, "\"RMSE\"") == 0)
        {
            appendStringInfo(&buf, "UPDATE public.%s_predict SET predict=%f "
                "WHERE row=%s;",
                tabname, result_pa[0],
                SPI_getvalue(spi_tuple, spi_tupdesc, 1));
        }
        else
        {
            double probability = sigmoid(result_pa[0]);
            int n = 0;

            if (probability > 0.5)
            {
                n = 1;
            }


            appendStringInfo(&buf,
                "UPDATE public.%s_predict "
                "SET predict=%f, class='%s' WHERE row=%s;",
                tabname, probability,(char*)*( modelClasses + n),
                SPI_getvalue(spi_tuple, spi_tupdesc, 1));
        }

        SPI_execute(buf.data, false, 0);

    }       // dataset
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

    text *filename = PG_GETARG_TEXT_PP(0);
    char *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));
    char* modelType;

    LoadModel(filename, &modelHandle);

    model_info = getModelParms(modelHandle);
    SPI_connect();
    modelType = getModelType(modelHandle, model_info);
    CretatePredictTable(tabname, modelType);
    modelClasses = getModelClasses(modelHandle, model_info);


    predict(modelHandle, tabname, NULL, modelType, modelClasses);


    if (SPI_finish() != SPI_OK_FINISH)
        elog(WARNING, "could not finish SPI");

    ModelCalcerDelete(modelHandle);

    initStringInfo(&buf);
    appendStringInfo(&buf, "public.%s_predict", tabname);

    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}



Datum
ml_predict_dataset_inner(PG_FUNCTION_ARGS)
{
    FuncCallContext *functionContext = NULL;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldContext;
        TupleDesc tupleDescriptor;
        ArrayDatum  cat_fields = {0, NULL};
        StringInfoData buf;
        ModelCalcerHandle* modelHandle;
        // SPITupleTable   *spi_tuptable;
        const char*     model_info;
        const int resultColumnCount = 3;
        int             model_float_feature_count;
        int             model_cat_feature_count ;
        int             model_dimension;
        size_t          featureCount = 0;
        char            **features;
        int             cat_feature_counter=0;
        int             feature_counter=0;
        int             i;
        MLmodel         model;
        text            *filename;
        char            *key_field;
        int             res;
        bool            function_type = PG_GETARG_BOOL(4);

        /* create a function context for cross-call persistence */
        functionContext = SRF_FIRSTCALL_INIT();

        filename = PG_GETARG_TEXT_PP(0);


        LoadModel(filename, &modelHandle);

        /* switch to memory context appropriate for multiple function calls */
        oldContext = MemoryContextSwitchTo(
            functionContext->multi_call_memory_ctx);

        initStringInfo(&buf);

        if (!PG_ARGISNULL(2))
        {

            int         key_count;
            Datum      *key_datums;
            bool       *key_nulls;
            ArrayType  *key_array = PG_GETARG_ARRAYTYPE_P(2);

            if (key_array == NULL) {
                elog(ERROR, "key is null");
            }

            Assert(ARR_ELEMTYPE(key_array) == TEXTOID);

            deconstruct_array(key_array,
                              TEXTOID, -1, false, TYPALIGN_INT,
                              &key_datums, &key_nulls, &key_count);

            cat_fields.elements = key_datums;
            cat_fields.count = key_count;
        }

        key_field = text_to_cstring(PG_GETARG_TEXT_PP(3));

        model = (MLmodel) palloc0(sizeof(MLmodelData));

        model->modelHandle = modelHandle;
        model_info = getModelParms(model->modelHandle);
        SPI_connect();
        model->modelType = getModelType(model->modelHandle, model_info);
        model->modelClasses = getModelClasses(model->modelHandle, model_info);

        if (key_field)
            model->keyField = pstrdup(key_field);

        if (function_type) {
            char  *query = text_to_cstring(PG_GETARG_TEXT_PP(1));
            res = SPI_exec(query, 0);
        }
        else
        {
            char  *tabname = text_to_cstring(PG_GETARG_TEXT_PP(1));
            appendStringInfo(&buf, "SELECT * FROM %s;", tabname);
            res = SPI_exec(buf.data, 0);
        }
        if (res < 1 || SPI_tuptable == NULL)
        {
            elog(ERROR, "Query %s error", buf.data);
        }

        model->cat_fields = cat_fields;
        model->spi_tuptable = SPI_tuptable;
        model->spi_tupdesc  = SPI_tuptable->tupdesc;
        model->attCount = SPI_tuptable->tupdesc->natts;


        model->iscategory = palloc0( model->attCount * sizeof(int8));
        model_float_feature_count = (int)GetFloatFeaturesCount(model->modelHandle);
        model_cat_feature_count = (int)GetCatFeaturesCount(model->modelHandle);
        model_dimension = (int)GetDimensionsCount(model->modelHandle);


        features = GetModelFeatures(model->modelHandle, &featureCount);

        for(i=0; i < model->attCount; i++)
        {
            if(! checkInArray(ModelGetFieldName(i), features, featureCount))
            {
                model->iscategory[i] = -1;  // not in features
                continue;
            }

            if ( checkInTextArray(ModelGetFieldName(i), &cat_fields) )
            {
                cat_feature_counter ++;
                model->iscategory[i] = 1;
            }
            else
            {
                feature_counter ++;
                model->iscategory[i] = 0;
            }
        }

         //  check model features
        if (feature_counter != model_float_feature_count)
        {
            elog(ERROR,
                "count of numeric features is not valid, must be %d is %d",
                model_float_feature_count, feature_counter
            );
        }

        if (cat_feature_counter != model_cat_feature_count)
        {
            elog(ERROR,
                "count of categocical features is not valid, must be %d is %d",
                model_cat_feature_count, cat_feature_counter
            );
        }

        model->current = 0;
        functionContext->user_fctx = model;
        functionContext->max_calls = SPI_processed;
        model->dimension = model_dimension;
        model->cat_count = cat_feature_counter;
        model->num_count = feature_counter;


        /*
         * This tuple descriptor must match the output parameters declared for
         * the function in pg_proc.
         */
        tupleDescriptor = CreateTemplateTupleDesc(resultColumnCount);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 1, key_field,
                           TEXTOID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 2, "predict",
                           FLOAT8OID, -1, 0);
        TupleDescInitEntry(tupleDescriptor, (AttrNumber) 3, "class",
                           TEXTOID, -1, 0);

        functionContext->tuple_desc =  BlessTupleDesc(tupleDescriptor);

        MemoryContextSwitchTo(oldContext);
    }

    functionContext = SRF_PERCALL_SETUP();

    if (  ((MLmodel) functionContext->user_fctx)->current < functionContext->max_calls)
    {
        char*   class;
        int     feature_counter = 0;
        int     cat_feature_counter = 0;
        MLmodel model = (MLmodel)functionContext->user_fctx;
        Datum   recordDatum;
        int j;
        char *p;
        char* yes = "yes";
        char* no = "no";
        HeapTuple   spi_tuple = ((MLmodel)functionContext->user_fctx)->spi_tuptable->vals[((MLmodel)functionContext->user_fctx)->current];
        int memsize = model->cat_count * sizeof(char) * FIELDLEN;
        char* key_field_value = NULL;


        model->row_fvalues = palloc0( model->num_count * sizeof(float));
        p = model->cat_value_buffer  = palloc0(memsize);
        model->row_cvalues = palloc0(model->cat_count * sizeof(char*));

        model->result_pa  = (double*) palloc( sizeof(double) * model->dimension);
        model->result_exp = (double*) palloc( sizeof(double) * model->dimension);

        for(j=0; j < model->attCount; j++)
        {
            char    *value;
            value = SPI_getvalue(spi_tuple, model-> spi_tupdesc, j+1);

            if (strcmp(model->keyField,  ModelGetFieldName(j)) == 0)
            {
                key_field_value = value;
            }

            if( model->iscategory[j] == -1) // not in features
                continue;

            if( model->iscategory[j] == 0)
            {
                if (value == 0 )
                {
                    model->row_fvalues[feature_counter] = QNaN;
                }
                else
                {
                    int res;
                    res  = sscanf(value, "%f", &model->row_fvalues[feature_counter]);
                    if(res < 1)
                    {
                        elog(WARNING,"error input j/cnt=%d/%d %f\n", j,
                            feature_counter, model->row_fvalues[feature_counter]);
                    }
                }

                feature_counter++;
            }

            if( model->iscategory[j] == 1)
            {
                if (!value)
                {
                    model->row_cvalues[cat_feature_counter] = pstrdup("NaN");
                    p += 3;
                }
                else
                {
                    model->row_cvalues[cat_feature_counter] = strcpy(p, value);
                    p += strlen(value);
                }
                cat_feature_counter++;
                *p = '\0';
                p++;
            }
        } // column

        if (!CalcModelPredictionSingle(model->modelHandle,
                    model->row_fvalues, feature_counter,
                    (const char** )model->row_cvalues, cat_feature_counter,
                    model->result_pa, model->dimension)
            )
        {
            StringInfoData str;
            initStringInfo(&str);
            for(j = 0; j < feature_counter; j++)
            {
                appendStringInfo(&str, "%f,", model->row_fvalues[j]);
            }
            elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%ld",
                    GetErrorString(),  model->current );
        }

        if (strncmp("\"MultiClass\"", model->modelType, 12) == 0)
        {
            char  ***p;
            double max = 0., sm = 0.;
            int max_i = -1;
            char* out = model->keyField;

            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = exp(model->result_pa[j]);
                sm += model->result_exp[j];
            }
            for( j = 0; j < model->dimension; j ++)
            {
                model->result_exp[j] = model->result_exp[j] / sm;
                if (model->result_exp[j] > max){
                    max = model->result_exp[j];
                    max_i = j;
                }
            }

            p = model->modelClasses + max_i;

            if (key_field_value)
            {
                out = key_field_value;
            }


            recordDatum = PredictGetDatum(out, model->current, max, (char*)*p,
                            functionContext->tuple_desc);

        }
        else if (strcmp(model->modelType, "\"RMSE\"") == 0)
        {
            char* out = "";

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, model->result_pa[0], NULL,
                            functionContext->tuple_desc);

        }
        else if (strncmp("\"Logloss\"", model->modelType, 9) == 0)
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            int n = 0;
            if (probability > 0.5)
            {
                n = 1;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }
            recordDatum = PredictGetDatum(out, model->current, probability,
                            (char*)*(model->modelClasses + n),
                            functionContext->tuple_desc);
        }
        else
        {
            double probability = sigmoid(model->result_pa[0]);
            char* out = model->keyField;
            if (probability > 0.5)
            {
                class=yes;
            }
            else
            {
                class=no;
            }

            if (key_field_value)
            {
                out = key_field_value;
            }

            recordDatum = PredictGetDatum(out, model->current,
                                          probability, class,
                                          functionContext->tuple_desc);

        }


        ((MLmodel) functionContext->user_fctx)->current++;

        pfree(model->row_fvalues);
        pfree(model->cat_value_buffer);
        pfree(model->row_cvalues);

        pfree(model->result_pa);
        pfree(model->result_exp);


        SRF_RETURN_NEXT(functionContext, recordDatum);
    }
    else
    {
        MLmodel model = (MLmodel)functionContext->user_fctx;
        pfree(model->keyField);

        if (SPI_finish() != SPI_OK_FINISH)
            elog(WARNING, "could not finish SPI");

        SRF_RETURN_DONE(functionContext);
    }
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

    appendStringInfo(&buf,
        "SELECT '%s'::jsonb #> '{loss_function,type}';", info);

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
    appendStringInfo(&buf,
        "SELECT '%s'::jsonb #> '{data_processing_options,class_names}';",
        info);

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

        res = (char***) palloc( sizeof(char*) * (
                getJsonbLength((const JsonbContainer*) j,0) + 1)
              );
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
                   continue;
                }
                if(jb.type == jbvNumeric)
                {
                    Numeric num;
                    num = jb.val.numeric;

                    *p = (char**)pstrdup(DatumGetCString(
                                            DirectFunctionCall1(numeric_out,
                                              NumericGetDatum(num))));
                    p++;
                   continue;
                }
                elog(ERROR, "undefined jsonb type num=%d",jb.type);
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

    appendStringInfo(&buf,
        "dimension:%ld numeric features:%ld categorial features:%ld",
        GetDimensionsCount(modelHandle),
        GetFloatFeaturesCount(modelHandle),
        GetCatFeaturesCount(modelHandle));

    model_info = getModelParms(modelHandle);

    SPI_connect();
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
ml_json_parms_info(PG_FUNCTION_ARGS)
{
    MemoryContext oldcontext;
    FILE *model = NULL;
    struct stat stat_info;
    char *buf = NULL;
    size_t readed = 0;
    size_t count_bytes = 0;
    char  *p;
    int   err;
    StringInfoData query;
    StringInfoData out;
    TupleDesc tupdesc;
    SPITupleTable *tuptable;
    int ret, i;
    StringInfoData sbuf;
    char slash[2] = "/\0";

    const char  *filename_str = text_to_cstring(PG_GETARG_TEXT_PP(0));
    oldcontext = MemoryContextSwitchTo(ml_context);


    initStringInfo(&sbuf);
    if (strstr(filename_str, slash) == NULL)
    {
        int len = strlen( model_path);
        if (model_path[len-1] == '/')
            appendStringInfo(&sbuf, "%s%s", model_path, filename_str);
        else
            appendStringInfo(&sbuf, "%s/%s", model_path, filename_str);

        filename_str = sbuf.data;
    }


    if (stat(filename_str, &stat_info) == -1)
    {
        err = errno;
        elog(ERROR, "file %s has error: %d:%s",filename_str, err, strerror(err));
    }

    if (stat_info.st_size < 16)
    {
        elog(ERROR, "file %s has very small size: %ld",filename_str,
            stat_info.st_size);
    }

    model = fopen(filename_str, PG_BINARY_R);
    if (model == NULL){
        err = errno;
        elog(ERROR, "Cannot open model file \"%s\": %s",
             filename_str, strerror(err));
    }

    buf = (char*)palloc0(stat_info.st_size+1);
    if (!buf)
    {
        elog(ERROR, "the filesize %s is very big %ld", filename_str,
            stat_info.st_size);
    }

    p = buf;
    while (!feof(model)){
        readed = fread(p, 1, PAGESIZE, model);
        p += readed;
        count_bytes += readed;
    }

    fclose(model);
    if (count_bytes != stat_info.st_size)
    {
        err = errno;
        elog(ERROR, "error readed len=%ld/%ld from file \n%s", readed,
            stat_info.st_size,strerror(err));
    }


    initStringInfo(&query);
    appendStringInfo(&query,
        "SELECT json_array_elements(('%s'::jsonb #> '{features_info,float_features}')::json)"
        " #>> '{feature_id}'", buf);

    tuptable = SPI_tuptable;
    SPI_connect();

    ret = SPI_exec(query.data, 0);
    tuptable = SPI_tuptable;

    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tuptable = SPI_tuptable;
    if (tuptable == NULL)
        elog(ERROR,"tuptable is null");

    initStringInfo(&out);
    if (SPI_processed > 0)
    {
        tupdesc = tuptable->tupdesc;

        appendStringInfo(&out,"float feature:");
        for (i = 0; i < SPI_processed; i++)
        {
            appendStringInfo(&out, "%s,", SPI_getvalue(tuptable->vals[i], tupdesc, 1));
        }
        p = p + out.len - 1;
        *p = '\n';   // SigFall
    }
    else
    {
        appendStringInfo(&out,"none float feature\n");
    }


    resetStringInfo(&query);
    appendStringInfo(&query,
        "SELECT json_array_elements(('%s'::jsonb #> '{features_info,categorical_features}')::json)"
        " #>> '{feature_id}'", buf);


    tuptable = SPI_tuptable;

    ret = SPI_exec(query.data, 0);
    tuptable = SPI_tuptable;

    if (ret < 1 || tuptable == NULL)
    {
        elog(ERROR, "Query errorcode=%d", ret);
    }

    tuptable = SPI_tuptable;
    if (tuptable == NULL)
        elog(ERROR,"tuptable is null");

    tupdesc = tuptable->tupdesc;

    appendStringInfo(&out,"\ncategorial feature:");
    for (i = 0; i < SPI_processed; i++)
    {
        appendStringInfo(&out, "%s,", SPI_getvalue(tuptable->vals[i], tupdesc, 1));
    }
    *(p + out.len - 1) = '\n';

    SPI_finish();

    pfree(buf);
    pfree(query.data);

    /* Reset our memory context and switch back to the original one */
    MemoryContextSwitchTo(oldcontext);
    MemoryContextReset(ml_context);

    PG_RETURN_TEXT_P(cstring_to_text(out.data));
}



static Datum
PredictGetDatum(char* id, int64 row_no, float8 predict, char* className,
                TupleDesc tupleDescriptor)
{
    Datum values[3];
    bool isNulls[3];
    HeapTuple htuple;

    memset(values, 0, sizeof(values));
    memset(isNulls, false, sizeof(isNulls));


    if ( strncmp("row", id, 3) )
    {
        values[0] = CStringGetTextDatum(id);
    }
    else
    {
        char row_str[MAXDIGIT];
        pg_itoa(row_no, row_str); 
        values[0] = CStringGetTextDatum(row_str);
    }

    values[1] = Float8GetDatum(predict);

    if (className) {
        values[2] =  CStringGetTextDatum(className);
    }
    else
    {
        isNulls[2] = true;
        values[2] =  CStringGetTextDatum("");
    }
    
    htuple = heap_form_tuple(tupleDescriptor, values, isNulls);
    return (Datum) HeapTupleGetDatum(htuple);
}


/*
 * see ReceiveResults(WorkerSession *session, bool storeRows)
 *
 */
Datum
ml_test(PG_FUNCTION_ARGS)
{



    PG_RETURN_TEXT_P(NULL);
}

/*
 * Check existing model folder
 *
 */
static bool
check_model_path(char **newval, void **extra, GucSource source)
{
    struct stat st;

    /*
     * The default value is an empty string, so we have to accept that value.
     * Our check_configured callback also checks for this and prevents
     * archiving from proceeding if it is still empty.
     */
    if (*newval == NULL || *newval[0] == '\0')
        return true;

    /*
     * Make sure the file paths won't be too long.  The docs indicate that the
     * file names to be archived can be up to 64 characters long.
     */
    if (strlen(*newval) + 64 + 2 >= MAXPGPATH)
    {
        GUC_check_errdetail("directory too long.");
        return false;
    }

    /*
     * Do a basic sanity check that the specified archive directory exists. It
     * could be removed at some point in the future, so we still need to be
     * prepared for it not to exist in the actual archiving logic.
     */
    if (stat(*newval, &st) != 0 || !S_ISDIR(st.st_mode))
    {
        GUC_check_errdetail("Specified  directory does not exist.");
        return false;
    }

    return true;
}


/*
 * _PG_init
 *
 * Defines the module's GUC.
 */
void
_PG_init(void)
{

    DefineCustomStringVariable("ml.model_path",
                               "Path to model folder",
                               NULL,
                               &model_path,
                               "",
                               PGC_SIGHUP,
                               0,
                               check_model_path, NULL, NULL);

    MarkGUCPrefixReserved("ml");

    ml_context = AllocSetContextCreate(TopMemoryContext,
                                       "ml",
                                       ALLOCSET_DEFAULT_SIZES); // 8K -> 8M
}
