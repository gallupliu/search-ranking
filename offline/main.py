from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType
from offline.utils.utils import CreateSparkContex
from offline.utils.utils import Segmentation, clean_text
from offline.models.TFIDF import TFIDF
from offline.models.TextRank import textrank

if __name__ == "__main__":
    sc, spark = CreateSparkContex()
    data = [
        ("牛奶", "1", "【3月新货】伊利金典纯牛奶 250ml*24盒整箱营养早餐牛奶", None, 0.8, 1, "[4,3]", [4, 3], ['牛奶', '奶制品'], "A", "A"),
        ("牛奶", "2", "蒙牛 精选特仑苏纯牛奶250ml*12盒金牌牛奶", None, 0.5, 1, "[3]", [3], ['牛奶', '奶制品'], "A", "A"),
        ("牛奶", "3", "【镇店之宝】蒙牛 纯牛奶250ml*12盒牛奶整箱【4月新货】", 100, 0.6, 2, None, None, ['牛奶', '奶制品'], "B", "A"),
        ("高血压", "4", "傲坦 奥美沙坦酯片", 0, 0.3, 3, "[2, 3, 4]", [], ['牛奶', '奶制品'], "C", "A"),
        ("口罩", "5", "[下单多送10片口罩]连花 连花一次性使用医用口罩50片/盒三层含熔喷布一次性医用承认口罩", 10, 0.9, 3, "[1, 2, 3, 4]", [1, 2, 3, 4],
         ['牛奶', '奶制品'],
         "D", "E"),
        ("高血压", "6", "【自营正品】吉加 厄贝沙坦片 0.15g*7片", 20, 0.8, 5, "[2]", [2], ['牛奶', '奶制品'], "D", "E"),
        ("口罩", "7", "索蜜 医用外科口罩索蜜一次性医用无菌平面型耳挂式口罩", 50, 0.78, 4, "[1, 2, 3]", [1, 2, 3], ['牛奶', '奶制品'], "E", "E"),
        ("避孕药", "8", "优思明 屈螺酮炔雌醇片3mg：0.03mg*21s", 20, 0.8, 5, "[1, 2]", [1, 2], ['牛奶', '奶制品'], "E", "E"),
        ("避孕药", "9", "达英-35 屈螺酮炔雌醇片21s", 50, 0.7, 6, "[1, 2, 3]", [1, 2, 3], ['牛奶', '奶制品'], "B", "E"),
        ("避孕套", "10", "【买二送一】OLO 玻尿酸0.01 避孕套超薄001热感 男用持久冰火两重天一体套 套套超润滑安全套", None, 0.7, 6, "[1, 2, 3]", [1, 2, 3], None,
         "C",
         "E"),
        (
        "避孕套", "11", "双11 双11 避孕套玻尿酸润滑剂安全套琅琊颗粒立体大颗粒型套套成人情趣用品男用", 50, 0.7, None, "[1, 2, 3]", [1, 2, 3], None, "E", "E"),
    ]
    df = spark.createDataFrame(data,
                               ["query", "id", "title", "price", "rate", "category_id", "tag_ids", "ids", "tag_texts",
                                "category", "test"]).select(["query", "id", "title", "price", "rate",  "tag_texts"])

    seg_tool = Segmentation('./data/goods/user_dict.txt', './data/goods/stopwords.txt')
    # add_embedding = udf(seg_tool.cut_sentence, ArrayType(StringType()))
    #
    # df = df.withColumn('title' + '_charvec', add_embedding('title'))
    df = seg_tool.cut(df, 'title')
    tmp = ['title_clean', 'title_list']
    df.select(*tmp).show()
    tfidf = TFIDF(df, 'title_list', './data/models/')
    tfidf_result = tfidf.train_model()
    print('tfidf')
    tfidf_result.show()

    keywords_list_with_idf = list(zip(tfidf.cv_model.vocabulary, tfidf.idf_model.idf.toArray()))


    def func(data):
        for index in range(len(data)):
            data[index] = list(data[index])
            data[index].append(index)
            data[index][1] = float(data[index][1])


    func(keywords_list_with_idf)
    sc = spark.sparkContext
    rdd = sc.parallelize(keywords_list_with_idf)
    idf_keywords_values = rdd.toDF(["keyword", "idf", "index"])
    print('idf_keywords_values')
    idf_keywords_values.show()
    idf_keywords_values.registerTempTable("idf_keywords_values")

    # 利用结果索引与”idf_keywords_values“合并知道词
    keywordsByTFIDF = tfidf_result.join(idf_keywords_values, ['index'])
    print('keywordsByTFIDF')
    keywordsByTFIDF.show()
    # keywordsByTFIDF.write.insertInto("tfidf_keywords_values")
    keywordsByTFIDF.registerTempTable("tfidf_keywords_values")
    # # 计算textrank
    textrank_keywords_df = df.rdd.mapPartitions(textrank(seg_tool.stopwords_list)).toDF(
        ["query", "id", "keyword", "textrank"])
    textrank_keywords_df.registerTempTable("textrank_keywords_values")
    print('textrank_keywords_df')
    textrank_keywords_df.show()

    # 加载IDF，保留关键词以及权重计算(TextRank * IDF)
    result = textrank_keywords_df.join(keywordsByTFIDF, ['query', 'id', 'keyword'])
    keywords_res = result.withColumn("weights", result.textrank * result.idf).select(
        ["query", "id", "keyword", "weights"])
    print('textrank*idf')
    keywords_res.show()

    # 合并关键词权重到字典结果
    keywords_res.registerTempTable("temptable")
    merge_keywords = spark.sql(
        "select id,collect_list(keyword) keywords,collect_list(weights) weights from temptable group by id")
    # df= df.join(merge_keywords,['id']).select(*['query','id','title','keywords','weights'])
    df = df.join(merge_keywords, ['id'])
    print('id')
    df.show()


    # merge_keywords.show()
    # 合并关键词权重合并成字典
    def _func(row):
        return row.id, row.query, row.title, row.title_list, dict(zip(row.keywords, row.weights))


    keywords_info = df.rdd.map(_func).toDF(["id", "query", "title", "title_list","keywords"])

    # 将tfidf和textrank共现的词作为主题词
    topic_sql = """
                    select t.id  id1, collect_set(t.keyword) topics from tfidf_keywords_values t
                    inner join
                    textrank_keywords_values r
                    where t.keyword=r.keyword
                    group by id1
                    """
    article_topics = spark.sql(topic_sql)
    article_topics = article_topics.withColumnRenamed('id1', 'id')
    article_topics.show()

    # 将主题词表和关键词表进行合并
    article_profile = keywords_info.join(article_topics, ['id'])
    article_profile.show()
    #
    # .select(
    #     ["article_id", "channel_id", "keywords", "topics"])

    # articleProfile.write.insertInto("article_profile")
