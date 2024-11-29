from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

# %% load the data
dat = "./dat"
user_artist = f"{dat}/user_artist_data_small.txt"
artist_alias = f"{dat}/artist_alias_small.txt"
artist_data = f"{dat}/artist_data_small.txt"

ua_dat     = open(user_artist)
aa_dat = open(artist_alias)

ua_lines, aa_lines = ua_dat.read().splitlines(), aa_dat.read().splitlines()
ua = []
aa = dict()

print("Preprocessing artist alias data...")
for line in aa_lines:
    line = line.split()
    aa[line[0]] = line[1]
    print(f"Line of aa: {line}")
print("Done preprocessing artist alias data...")

print("Preprocessing user artist data...")
for line in ua_lines:
    line = line.split()
    if line[1] in aa.keys():
        print(f"Line of ua before clean: {line}")
        line[1] = aa[line[1]]
        print(f"Line of ua after clean: {line}")
    line = [int(entry) for entry in line]

    ua.append(line)
print("Done preprocessing user artist data...")

# %% Start Spark session
spark = SparkSession.builder\
    .master("local")\
    .appName("Audioscrobbler Analyzer")\
    .getOrCreate()

schema = StructType([
    StructField("userid", IntegerType(), True),
    StructField("artistid", IntegerType(), True),
    StructField("playcount", IntegerType(), True),
])

ua_matrix = spark.createDataFrame(ua, schema)


# fn for computing person similarity
def compute_pearson_similarity(userid1, userid2, ua_matrix: DataFrame) -> float:
    user1 = (ua_matrix
             .select(["userid", "artistid", "playcount"])
             .where(col("userid")==userid1)
             .withColumnRenamed("playcount", "playcount1"))
    user2 = (ua_matrix
             .select(["userid", "artistid", "playcount"])
             .where(col("userid")==userid2)
             .withColumnRenamed("playcount", "playcount2"))
    user_joined = user1.join(user2, on="artistid", how="inner")
    return user_joined.corr(method='pearson', col1="playcount1", col2="playcount2")


# fn for computing knn
def compute_knn(user, ua_matrix: DataFrame, k):
    others = ua_matrix.select("userid").where(col("userid") != user).distinct()

    schema = StructType([
        StructField("userid", IntegerType(), True),
        StructField("other_userid", IntegerType(), True),
        StructField("similarity", FloatType(), True),
    ])

    similarities = spark.createDataFrame([], schema)

    for other in others.rdd.collect():
        similarity = compute_pearson_similarity(other[0], user, ua_matrix)
        similarity = spark.createDataFrame([(user, other[0], similarity)], schema)
        similarities = similarities.union(similarity)

    return similarities.dropna(how='any').sort("similarity", ascending=False).take(k)


print(compute_knn(1059637, ua_matrix, 10))