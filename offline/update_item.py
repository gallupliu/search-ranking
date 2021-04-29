import os
import sys
import json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
from datetime import datetime
from datetime import timedelta
import pyspark.sql.functions as pyf
import pyspark
import gc

from offline.utils.utils import Segmentation, clean_text
from offline.utils.utils import CreateSparkContex
from offline.models.TFIDF import TFIDF
from offline.models.TextRank import textrank


class UpdateItem(object):
    """
    更新item画像
    """
    SPARK_APP_NAME = "updateItem"
    ENABLE_HIVE_SUPPORT = True

    SPARK_EXECUTOR_MEMORY = "7g"

    def __init__(self, cv_path, idf_path, top_k):
        # self.spark = self._create_spark_session()

        self.cv_path = cv_path
        self.idf_path = idf_path
        self.top_k = top_k
        self.seg_tool = Segmentation('./data/goods/user_dict.txt', './data/goods/stopwords.txt')
        self.cv_model = self.get_cv_model()
        self.idf_model = self.get_idf_model()

    def get_cv_model(self):
        # 词语与词频统计
        from pyspark.ml.feature import CountVectorizerModel
        cv_model = CountVectorizerModel.load(self.cv_path)
        return cv_model

    def get_idf_model(self):
        from pyspark.ml.feature import IDFModel
        idf_model = IDFModel.load(self.idf_path)
        return idf_model

    def get_idf_keywords_values(self):
        keywords_list_with_idf = list(zip(self.cv_model.vocabulary, self.idf_model.idf.toArray()))

        def func(data):
            for index in range(len(data)):
                data[index] = list(data[index])
                data[index].append(index)
                data[index][1] = float(data[index][1])

        func(keywords_list_with_idf)
        sc = spark.sparkContext
        rdd = sc.parallelize(keywords_list_with_idf)
        idf_keywords_values = rdd.toDF(["word", "idf", "index"])
        return idf_keywords_values


    @staticmethod
    def compute_keywords_tfidf_topk(words_df, cv_model, idf_model, column, TOPK):
        """对指定列保存tfidf值高的topk个关键词
        :param spark:
        :param words_df:
        :return:
        """
        print('word_df')
        words_df.show()
        cv_result = cv_model.transform(words_df)
        tfidf_result = idf_model.transform(cv_result)

        # print("transform compelete")
        def func(partition):
            print('TFIDF TRAIN MODEL FUNC')
            for row in partition:
                # 找到索引与IDF值并进行排序
                print('row:{0}'.format(row))
                # print(column)
                _ = list(zip(row[column + 'idfFeatures'].indices, row[column + 'idfFeatures'].values))
                # _ = list(zip(row['title_listidfFeatures'].indices, row['title_listidfFeatures'].values))
                _ = sorted(_, key=lambda x: x[1], reverse=True)
                result = _[:TOPK]
                for word_index, tfidf in result:
                    yield row.query, row.id, int(word_index), row[column], round(float(tfidf), 4)

        tfidf_result.show()
        _keywordsByTFIDF = tfidf_result.rdd.mapPartitions(func).toDF(["query", "id", "index", column, column + "tfidf"])

        return _keywordsByTFIDF

    # def merge_item_data(self):
    #     """
    #     合并业务中增量更新的item数据
    #     :return:
    #     """
    #     # 获取item相关数据, 指定过去一个小时整点到整点的更新数据
    #     # 如：26日：1：00~2：00，2：00~3：00，左闭右开
    #     self.spark.sql("use toutiao")
    #     _yester = datetime.today().replace(minute=0, second=0, microsecond=0)
    #     start = datetime.strftime(_yester + timedelta(days=0, hours=-1, minutes=0), "%Y-%m-%d %H:%M:%S")
    #     end = datetime.strftime(_yester, "%Y-%m-%d %H:%M:%S")
    #
    #     # 合并后保留：article_id、channel_id、channel_name、title、content
    #     # +----------+----------+--------------------+--------------------+
    #     # | article_id | channel_id | title | content |
    #     # +----------+----------+--------------------+--------------------+
    #     # | 141462 | 3 | test - 20190316 - 115123 | 今天天气不错，心情很美丽！！！ |
    #     basic_content = self.spark.sql(
    #         "select a.article_id, a.channel_id, a.title, b.content from news_article_basic a "
    #         "inner join news_article_content b on a.article_id=b.article_id where a.review_time >= '{}' "
    #         "and a.review_time < '{}' and a.status = 2".format(start, end))
    #     # 增加channel的名字，后面会使用
    #     basic_content.registerTempTable("temparticle")
    #     channel_basic_content = self.spark.sql(
    #         "select t.*, n.channel_name from temparticle t left join news_channel n on t.channel_id=n.channel_id")
    #
    #     # 利用concat_ws方法，将多列数据合并为一个长文本内容（频道，标题以及内容合并）
    #     self.spark.sql("use article")
    #     sentence_df = channel_basic_content.select("article_id", "channel_id", "channel_name", "title", "content", \
    #                                                pyf.concat_ws(
    #                                                    ",",
    #                                                    channel_basic_content.channel_name,
    #                                                    channel_basic_content.title,
    #                                                    channel_basic_content.content
    #                                                ).alias("sentence")
    #                                                )
    #     del basic_content
    #     del channel_basic_content
    #     gc.collect()
    #
    #     sentence_df.write.insertInto("article_data")
    #     return sentence_df

    def merge_item_data(self, sentence_df, column):
        """
        生成item标签  tfidf, textrank
        :param sentence_df: 增量的item内容
        :return:
        """
        # 进行分词
        words_df = self.seg_tool.cut(sentence_df, 'title')
        # words_df = sentence_df.rdd.mapPartitions(segmentation).toDF(["article_id", "channel_id", "words"])

        # 1、保存所有的词的idf的值，利用idf中的词的标签索引
        # 工具与业务隔离
        _columnByTFIDF = UpdateItem.compute_keywords_tfidf_topk(words_df, self.cv_model, self.idf_model,
                                                                column + '_list', 5)
        idf_keywords_values = self.get_idf_keywords_values()
        idf_keywords_values.registerTempTable("idf_keywords_values")
        print('test')
        _columnByTFIDF.show()
        idf_keywords_values.show()
        idf_keywords_values.select(pyf.to_json(pyf.create_map('word', 'idf').alias("mapper")).alias("dict")).coalesce(
            1).write.format("json").option("header","false").mode("overwrite").save('./idf.json')
        #通过collect_set函数对数据进行转换
        # coin_new=idf_keywords_values.groupBy("word").agg(pyf.array_sort(pyf.collect_set("idf")).alias("idf_values"))

        idf_df = idf_keywords_values.select(['word','idf']).toPandas()

        idf_dict = {}
        # Open new json file if not exist it will create
        with open('./test_idf.json', 'w') as  fp:
            for i in range(len(idf_df)):
                idf_dict[idf_df['word'][i]] = idf_df['idf'][i]
             # write to json file
            js = json.dumps(idf_dict)
            fp.write(js)


        columnByTFIDF = _columnByTFIDF.join(idf_keywords_values, ['index'])

        keywordsIndex = spark.sql("select word, index  from idf_keywords_values")
        keywordsIndex.show()

        keywordsByTFIDF = columnByTFIDF.join(keywordsIndex, ["index", "word"]).select(
            ["query", "id", "word", column + '_list', column + '_list' + "tfidf"])
        print('keywordsByTFIDF')
        keywordsByTFIDF.show()
        # keywordsByTFIDF.write.insertInto("tfidf_keywords_values")

        del _columnByTFIDF
        gc.collect()
        print('sentence')
        sentence_df.show()
        # 计算textrank
        textrank_keywords_df = sentence_df.rdd.mapPartitions(textrank(self.seg_tool.stopwords_list)).toDF(
            ["query", "id", "keyword", "textrank"])

        # 加载IDF，保留关键词以及权重计算(TextRank * IDF)
        columnByTFIDF = columnByTFIDF.withColumnRenamed("word", "keyword")
        result = textrank_keywords_df.join(columnByTFIDF, ['query', 'id', 'keyword'])
        keywords_res = result.withColumn("weights", result.textrank * result.idf).select(
            ["query", "id", "keyword", "weights"])
        print('textrank*idf')
        keywords_res.show()

        # 合并关键词权重到字典结果
        keywords_res.registerTempTable("temptable")
        merge_keywords = spark.sql(
            "select id,collect_list(keyword) keywords,collect_list(weights) weights from temptable group by id")
        # df= df.join(merge_keywords,['id']).select(*['query','id','title','keywords','weights'])
        df = words_df.join(merge_keywords, ['id'])

        def _func(row):
            return row.id, row.query, row[column], row.title_list, dict(zip(row.keywords, row.weights))

        keywords_info = df.rdd.map(_func).toDF(["id", "query", "title", "title_list", "keywords"])

        columnByTFIDF.registerTempTable("tfidf_keywords_values")
        textrank_keywords_df.registerTempTable("textrank_keywords_values")
        # 将tfidf和textrank共现的词作为主题词
        topic_sql = """
                        select t.id  id1, collect_set(t.keyword) topics from tfidf_keywords_values t
                        inner join
                        textrank_keywords_values r
                        where t.keyword=r.keyword
                        group by id1
                        """
        item_topics = spark.sql(topic_sql)
        item_topics = item_topics.withColumnRenamed('id1', 'id')
        print('topic')
        item_topics.show()

        # 将主题词表和关键词表进行合并
        item_profile = keywords_info.join(item_topics, ['id'])
        item_profile.show()

        return item_profile, keywordsIndex

    def get_item_profile(self, textrank, keywordsIndex, column):
        """
        item画像主题词建立
        :param idf: 所有词的idf值
        :param textrank: 每个item的textrank值
        :return: 返回建立号增量item画像
        """
        print('textrank')
        textrank.show()
        textrank.select('keywords').coalesce(
            1).write.format("json").option("header", "false").mode("overwrite").save('./textrank.json')

        # returnitemProfile


if __name__ == '__main__':
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
            "避孕套", "11", "双11 双11 避孕套玻尿酸润滑剂安全套琅琊颗粒立体大颗粒型套套成人情趣用品男用", 50, 0.7, None, "[1, 2, 3]", [1, 2, 3], None, "E",
            "E"),
    ]
    sentence_df = spark.createDataFrame(data,
                                        ["query", "id", "title", "price", "rate", "category_id", "tag_ids", "ids",
                                         "tag_texts",
                                         "category", "test"]).select(
        ["query", "id", "title", "price", "rate", "tag_texts"])

    ua = UpdateItem('./data/models/CV.model', './data/models/IDF.model', 5)
    # sentence_df = ua.merge_item_data()
    columns = ['title']
    for column in columns:
        if sentence_df.rdd.collect():
            rank, idf = ua.merge_item_data(sentence_df, column)
            itemProfile = ua.get_item_profile(rank, idf, column)
