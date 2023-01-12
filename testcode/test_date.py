import datetime, time
from dateutil.relativedelta import relativedelta


def trans():
    # 时间戳转日期
    timestamp = 1670987512869
    time_local = time.localtime(timestamp / 1000)
    print(time_local)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    print(dt)
    # 日期转时间戳
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    print(timeArray)
    timeStamp = int(time.mktime(timeArray))
    print(timeStamp)
    # 当前时间的前半年
    time_now = time.time()
    print(time_now)
    today = datetime.date.today()
    print(today)
    font_6_time = today - relativedelta(months=6)
    print("font_6_time", font_6_time)
    print("==========")
    t1 = time.strptime(str(today), "%Y-%m-%d")
    now = int(time.mktime(t1))
    print(now)
    t2 = time.strptime(str(font_6_time), "%Y-%m-%d")
    font_6 = int(time.mktime(t2))
    print(font_6)
    print(now-font_6)
    print((now - font_6)/3600/24/30)


if __name__ == '__main__':
    trans()
    print(time.localtime(time.time()))
