from datetime import datetime
from zoneinfo import ZoneInfo



def get_market_status():

    # 本地时间
    local_now = datetime.now()


    # 美东时间
    ny_now = datetime.now(
        ZoneInfo("America/New_York")
    )


    weekday = ny_now.weekday()


    # 默认状态
    status = "休市"


    # 周一到周五判断
    if weekday < 5:

        current = (
            ny_now.hour * 60
            +
            ny_now.minute
        )


        open_time = 9 * 60 + 30
        close_time = 16 * 60


        premarket = 4 * 60

        after_hours = 20 * 60


        if open_time <= current < close_time:

            status = "🟢 正常交易中"


        elif premarket <= current < open_time:

            status = "🟡 盘前交易"


        elif close_time <= current < after_hours:

            status = "🟠 盘后交易"


        else:

            status = "🔴 休市"



    return {

        "local_time":
            local_now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),


        "ny_time":
            ny_now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),


        "status":
            status

    }