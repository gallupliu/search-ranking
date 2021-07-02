# #-*- coding:utf-8 -*-

from pyhanlp import *
document = "[补贴爆款]lancome/兰蔻 小黑瓶安瓶精华肌底液100ml"

print(HanLP.segment(document))
# document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
#     "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
#            "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
#            "严格地进行水资源论证和取水许可的批准。"
print("=" * 30 + "关键词提取" + "=" * 30)
print(HanLP.extractKeyword(document, 5))
print("-" * 70)
print(HanLP.extractSummary(document, 1))