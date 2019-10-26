from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
# 读取文件
gq = 'goodqueries.txt'
bq = 'badqueries.txt'

def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r', encoding = 'utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines
positive = ReadTxtName(gq)
negative = ReadTxtName(bq)

# 符号区分
normal_symbol = ['+', '=', '%', '&', '#', '$', '!', '/', '.', '>', '?', '@', '^', '~']

# 文本信息重新构建
def rebuild(data, normal_symbol):
    container = []
    for item in data:
        str_ = ''
        for s in item:
            if s in normal_symbol:
                str_ += ' '
            else:
                str_ += s
        container.append(str_)
    return container

negative_ = rebuild(negative, normal_symbol)
positive_ = rebuild(positive, normal_symbol)

def rank(data):
    rank_lis = []
    split_lis = []
    for item in data:
        rank_lis += item.split()
        split_lis.append(item.split())
    rank_lis = sorted(Counter(rank_lis).items(), key = lambda x:x[1], reverse = True)
    return rank_lis, split_lis

negative_rank, negative_split = rank(negative_)
positive_rank, positive_split = rank(positive_)

def high_freq_word_extract(rank_lis):
    container = []
    for words in rank_lis:
        if words[1] >= 10:
            container.append(words[0])
    return container

negative_words = high_freq_word_extract(negative_rank)
positive_words = high_freq_word_extract(positive_rank)

# 训练集处理
def preprocessing(split_lis, words):
    container = []
    for item in split_lis:  
        temp = []
        for word in item:
            if word in words:
                temp.append(word)
            else:
                for i in range(len(word) - 2):
                    temp.append(word[i: i+3])
        container.append(' '.join(temp))
    return container
pos_keywords = preprocessing(positive_split, positive_words)
neg_keywords = preprocessing(negative_split, negative_words)

# 训练/测试集划分
num1 = int(len(pos_keywords) * 0.7)
num2 = int(len(neg_keywords) * 0.7)
train = pos_keywords[:num1] + neg_keywords[:num2]
pos_label = [-1 for i in range(num1)]
neg_label = [1 for i in range(num2)]
train_labels = pos_label + neg_label

num1 = int(len(pos_keywords) * 0.3)
num2 = int(len(neg_keywords) * 0.3)
test = pos_keywords[-num1:] + neg_keywords[-num2:]
pos_label = [-1 for i in range(num1)]
neg_label = [1 for i in range(num2)]
test_labels = pos_label + neg_label

# 模型构建
transformer = TfidfTransformer()
vectorizer = CountVectorizer()

x = vectorizer.fit_transform(train)
tfidf = transformer.fit_transform(x)

svm = LinearSVC(class_weight = 'balanced', random_state = 66)
svm.fit(tfidf, train_labels)

# 测试
x_ = vectorizer.transform(test)
tfidf_ = transformer.transform(x_)

predictions_ = svm.predict(tfidf_)

print(f1_score(test_labels, predictions_), recall_score(test_labels, predictions_))