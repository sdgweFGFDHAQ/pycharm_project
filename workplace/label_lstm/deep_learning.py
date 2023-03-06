import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool
from ast import literal_eval
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from global_parameter import StaticParameter as SP
from mini_tool import set_jieba, cut_word, error_callback
# from workplace.label_lstm.global_parameter import StaticParameter as SP
# from workplace.label_lstm.mini_tool import set_jieba, cut_word, error_callback
import gc

warnings.filterwarnings("ignore", category=UserWarning)


# 批量标准化
def get_city(city_list, i):
    for city in city_list:
        set_file_standard_data(city, i)


# 读取原始文件,将数据格式标准化
def set_file_standard_data(city, part_i):
    path_city = SP.PATH_ZZX_DATA + city + '.csv'
    path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv'
    if os.path.exists(path_city):
        csv_data = pd.read_csv(path_city, usecols=['id', 'name', 'category1_new', 'category2_new', 'category3_new'])
        # 用一级标签填充空白(NAN)的二级标签、三级标签
        # csv_data = csv_data[csv_data['category1_new'].notnull() & (csv_data['category1_new'] != "")]
        csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
        csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
        # 得到标准数据
        set_jieba()
        csv_data['cut_name'] = csv_data['name'].apply(cut_word)
        if os.path.exists(path_part) and not os.path.getsize(path_part):
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=['id', 'name', 'category3_new', 'cut_name'], mode='a', header=False)
        else:
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=['id', 'name', 'category3_new', 'cut_name'], mode='w')


def random_get_trainset():
    standard_df = pd.DataFrame(columns=['id', 'name', 'category3_new', 'cut_name'])
    for i in range(SP.SEGMENT_NUMBER):
        path = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(i) + '.csv'
        df_i = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'], keep_default_na=False)
        df_i = df_i[df_i['category3_new'] != '']
        # 全量数据
        # standard_df_i = df_i
        # 部分数据
        standard_df_i = df_i.groupby(df_i['category3_new']).sample(frac=0.15, random_state=23)
        standard_df = pd.concat([standard_df, standard_df_i])
        standard_df = standard_df.sample(frac=1).reset_index(drop=True)
    standard_df.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_data.csv', index=False)


def get_dataset():
    gz_df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_data.csv')
    # gz_df = pd.read_csv('standard_store_data.csv')
    print(len(gz_df.index))
    gz_df['cat_id'] = gz_df['category3_new'].factorize()[0]
    cat_df = gz_df[['category3_new', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    print(len(cat_df.index))
    cat_df.to_csv('category_to_id.csv')
    ic_dict = dict(zip(cat_df['cat_id'], cat_df['category3_new']))
    return gz_df, ic_dict


def fit_model_by_deeplearn(df):
    tokenizer = Tokenizer(num_words=SP.MAX_WORDS_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    sample_lists = list()
    for i in df['cut_name']:
        i = literal_eval(i)
        sample_lists.append(' '.join(i))
    tokenizer.fit_on_texts(sample_lists)
    word_index = tokenizer.word_index
    # print(word_index)
    print('共有 %s 个不相同的词语.' % len(word_index))
    X = tokenizer.texts_to_sequences(sample_lists)
    # 填充X,让X的各个列的长度统一
    X = pad_sequences(X, maxlen=SP.MAX_LENGTH)
    # # 多类标签的onehot展开
    Y = pd.get_dummies(df['cat_id']).values
    # 拆分训练集和测试集
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # 定义模型
    model = Sequential()
    model.add(Embedding(SP.MAX_WORDS_NUM, SP.EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, Y, epochs=5, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    # accuracy = model.evaluate(X_test, Y_test)
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0], accuracy[1]))
    return tokenizer, model


def draw_trend(history):
    # 绘制损失函数趋势图
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # 绘制准确率趋势图
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


def predict_result(tokenizer, model, id_cat_dict, part_i):
    try:
        df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv')
        test_lists = list()
        for i in df['cut_name']:
            i = literal_eval(i)
            test_lists.append(' '.join(i))
        seq = tokenizer.texts_to_sequences(test_lists)
        padded = pad_sequences(seq, maxlen=SP.MAX_LENGTH)
        pred_lists = model.predict(padded)
        print(pred_lists[:10])
        id_lists = pred_lists.argmax(axis=1)
        cat_lists = list()
        for id in id_lists:
            cat_lists.append(id_cat_dict[id])
        result = pd.DataFrame(
            {'store_id': df['id'], 'name': df['name'], 'category3_new': df['category3_new'],
             'predict_category': cat_lists})
        result.to_csv(SP.PATH_ZZX_PREDICT_DATA + 'predict_category_' + str(part_i) + '.csv')
        # result.to_csv('test_predict_category.csv')
    except Exception as e:
        with open('error_city.txt', 'a') as ef:
            ef.write('出错的city: ' + str(part_i) + '; 异常e:' + str(e))


# 用于重新切分店名，生成标准文件
def rerun_get_file():
    cities = ['江门市', '新乡市', '河源市', '潮州市', '湛江市', '肇庆市', '开封市', '广州市', '安阳市', '茂名市', '南阳市', '焦作市',
              '漯河市', '深圳市', '韶关市', '驻马店市', '商丘市', '汕头市', '许昌市', '揭阳市', '郑州市', '汕尾市', '惠州市', '平顶山市',
              '清远市', '济源市', '洛阳市', '周口市', '云浮市', '珠海市', '三门峡市', '鹤壁市', '信阳市', '佛山市', '梅州市', '濮阳市',
              '徐州市', '宿迁市', '无锡市', '盐城市', '泰州市', '齐齐哈尔市', '常州市', '黑河市', '大庆市', '镇江市', '扬州市', '鸡西市',
              '苏州市', '七台河市', '大兴安岭地区', '南通市', '鹤岗市', '南京市', '牡丹江市', '佳木斯市', '绥化市', '伊春市', '淮安市',
              '双鸭山市', '连云港市', '哈尔滨市', '随州市', '恩施土家族苗族自治州', '武汉市', '宜昌市', '杭州市', '黄冈市', '台州市',
              '温州市', '咸宁市', '鄂州市', '荆门市', '襄阳市', '舟山市', '神农架林区', '宁波市', '丽水市', '黄石市', '孝感市', '十堰市',
              '天门市', '荆州市', '仙桃市', '湖州市', '潜江市', '定安县', '本溪市', '辽阳市', '屯昌县', '朝阳市', '铁岭市', '锦州市',
              '阜新市', '儋州市', '临高县', '白沙黎族自治县', '鞍山市', '文昌市', '海口市', '陵水黎族自治县', '保亭黎族苗族自治县',
              '乐东黎族自治县', '琼海市', '葫芦岛市', '澄迈县', '万宁市', '五指山市', '三亚市', '丹东市', '抚顺市', '大连市', '益阳市',
              '昌江黎族自治县', '沈阳市', '三沙市', '北京城区', '营口市', '东方市', '盘锦市', '琼中黎族苗族自治县', '景德镇市',
              '黔南布依族苗族自治州', '中卫市', '南昌市', '石嘴山市', '贵阳市', '黔东南苗族侗族自治州', '九江市', '吴忠市', '六盘水市',
              '黔西南布依族苗族自治州', '上饶市', '抚州市', '银川市', '新余市', '毕节市', '吉安市', '遵义市', '铜仁市', '安顺市', '宜春市',
              '鹰潭市', '固原市', '萍乡市', '赣州市', '滨州市', '潍坊市', '聊城市', '济宁市', '济南市', '青岛市', '东营市', '威海市',
              '枣庄市', '烟台市', '菏泽市', '泰安市', '临沂市', '淄博市', '德州市', '日照市', '乌兰察布市', '保山市', '呼伦贝尔市',
              '鄂尔多斯市', '普洱市', '玉溪市', '临沧市', '三明市', '漳州市', '呼和浩特市', '曲靖市', '龙岩市', '迪庆藏族自治州', '通辽市',
              '楚雄彝族自治州', '宁德市', '泉州市', '阿拉善盟', '大理白族自治州', '南平市', '文山壮族苗族自治州', '丽江市', '包头市',
              '西双版纳傣族自治州', '乌海市', '昭通市', '怒江傈僳族自治州', '莆田市', '巴彦淖尔市', '厦门市', '德宏傣族景颇族自治州', '昆明市',
              '红河哈尼族彝族自治州', '兴安盟', '福州市', '赤峰市', '锡林郭勒盟', '澳门', '黄山市', '淮北市', '六安市', '宣城市', '合肥市',
              '铜陵市', '宿州市', '滁州市', '蚌埠市', '马鞍山市', '亳州市', '芜湖市', '阜阳市', '池州市', '安庆市', '淮南市', '沧州市',
              '保定市', '衡水市', '邢台市', '廊坊市', '邯郸市', '承德市', '秦皇岛市', '张家口市', '唐山市', '石家庄市', '铜川市',
              '榆林市', '渭南市', '延安市', '汉中市', '宝鸡市', '安康市', '西安市', '咸阳市', '商洛市', '玉树藏族自治州', '海东市',
              '巴中市', '辽源市', '延边朝鲜族自治州', '四平市', '遂宁市', '凉山彝族自治州', '海西蒙古族藏族自治州', '绵阳市', '海北藏族自治州',
              '泸州市', '白山市', '达州市', '眉山市', '阿坝藏族羌族自治州', '吉林市', '黄南藏族自治州', '内江市', '海南藏族自治州', '成都市',
              '广安市', '自贡市', '通化市', '长春市', '白城市', '南充市', '乐山市', '德阳市', '资阳市', '甘孜藏族自治州', '攀枝花市',
              '宜宾市', '松原市', '广元市', '雅安市', '果洛藏族自治州', '西宁市', '东莞市', '中山市', '湘潭市', '百色市', '玉林市',
              '怀化市', '防城港市', '河池市', '梧州市', '岳阳市', '郴州市', '钦州市', '崇左市', '常德市', '株洲市', '北海市', '柳州市',
              '桂林市', '张家界市', '娄底市', '永州市', '湘西土家族苗族自治州', '长沙市', '来宾市', '衡阳市', '邵阳市', '南宁市', '兰州市',
              '甘南藏族自治州', '金昌市', '酒泉市', '张掖市', '白银市', '嘉峪关市', '武威市', '天水市', '庆阳市', '临夏回族自治州',
              '陇南市', '平凉市', '定西市', '忻州市', '吕梁市', '阳泉市', '太原市', '长治市', '运城市', '临汾市', '晋城市', '晋中市',
              '贵港市', '贺州市', '朔州市', '大同市', '上海城区', '日喀则市', '五家渠市', '昌吉回族自治州', '那曲市', '阿里地区',
              '胡杨河市', '石河子市', '北屯市', '克拉玛依市', '克孜勒苏柯尔克孜自治州', '乌鲁木齐市', '山南市', '阿克苏地区',
              '博尔塔拉蒙古自治州', '吐鲁番市', '哈密市', '阿拉尔市', '双河市', '可克达拉市', '林芝市', '铁门关市', '喀什地区', '塔城地区',
              '天津城区', '伊犁哈萨克自治州', '拉萨市', '和田地区', '巴音郭楞蒙古自治州', '阿勒泰地区', '昆玉市', '图木舒克市', '昌都市',
              '重庆郊县', '重庆城区', '香港', '阳江市', '金华市', '嘉兴市', '衢州市', '绍兴市']
    # 初始化，清空文件
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_sta = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(csv_i) + '.csv'
        if os.path.exists(path_sta):
            open(path_sta, "r+").truncate()
    # 合并城市,分为12部分
    city_num = len(cities)
    index_list = [int(city_num * i / SP.SEGMENT_NUMBER) for i in range(SP.SEGMENT_NUMBER + 1)]
    pool = Pool(processes=4)
    for index in range(len(index_list) - 1):
        cities_i = cities[index_list[index]:index_list[index + 1]]
        pool.apply_async(get_city, args=(cities_i, index), error_callback=error_callback)
    pool.close()
    pool.join()


# 用于重新预测打标，生成预测文件
def rerun_get_model():
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_pre = SP.PATH_ZZX_PREDICT_DATA + 'predict_category_' + str(csv_i) + '.csv'
        if os.path.exists(path_pre):
            open(path_pre, "r+").truncate()
    # 训练模型,获取训练集
    random_get_trainset()
    df, id_cat_dict = get_dataset()
    tokenizer, model = fit_model_by_deeplearn(df)
    # 预测数据
    for i in range(SP.SEGMENT_NUMBER):
        predict_result(tokenizer, model, id_cat_dict, i)


if __name__ == '__main__':
    # 用于重新切分店名，生成标准文件
    rerun_get_file()
    # 随机抽取带标签训练集
    random_get_trainset()
    # 用于重新预测打标，生成预测文件
    # rerun_get_model()
    # 绘制收敛次数图像
    # draw_trend(model_fit)
# nohup python -u main.py > log.log 2>&1 &
