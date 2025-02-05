from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from prophecy.libs import typed_lit
from prophecy.transpiler import call_spark_fcn
from prophecy.transpiler.fixed_file_schema import *
from web_vectorize.config.ConfigStore import *
from web_vectorize.udfs.UDFs import *

def with_id(spark: SparkSession, FlattenSchema_1: DataFrame) -> DataFrame:
    return FlattenSchema_1.select(
        concat(lit("web-"), monotonically_increasing_id()).alias("id"), 
        col("url"), 
        col("content_chunk")
    )
