from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from prophecy.libs import typed_lit
from prophecy.transpiler import call_spark_fcn
from prophecy.transpiler.fixed_file_schema import *
from web_vectorize.config.ConfigStore import *
from web_vectorize.udfs.UDFs import *

def vector_db(spark: SparkSession, web_silver_content_vectorized_1: DataFrame):
    from pyspark.sql.functions import expr, array, struct
    from spark_ai.dbs.pinecone import PineconeDB, IdVector
    from pyspark.dbutils import DBUtils
    PineconeDB(DBUtils(spark).secrets.get(scope = "rj-pinecone", key = "api_key"), "us-east-1-aws").register_udfs(spark)
    web_silver_content_vectorized_1\
        .withColumn("_row_num", row_number().over(Window.partitionBy().orderBy(col("id"))))\
        .withColumn("_group_num", ceil(col("_row_num") / 20))\
        .withColumn("_id_vector", struct(col("id"), col("embedding").alias("vector")))\
        .groupBy(col("_group_num"))\
        .agg(collect_list(col("_id_vector")).alias("id_vectors"))\
        .withColumn("upserted", expr(f"pinecone_upsert(\"rj-index\", id_vectors)"))\
        .select(col("*"), col("upserted.*"))\
        .select(col("id_vectors"), col("count"), col("error"))\
        .count()
