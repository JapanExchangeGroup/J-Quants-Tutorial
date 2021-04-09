import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# バックテスト実行用クラス
class Backtest(object):
    # バックテストパラメータ
    # 対象列定義
    DATE = "date"
    CODE = "Local Code"
    BUDGET = "budget"
    # 日付フォーマット定義
    DATE_FORMAT = "%Y-%m-%d"
    # 購入順列 (読み込み時にレコード順を付与する)
    BUYING_ORDER = "n"
    # 最低銘柄選択数。これより少ない場合はエラーとする
    MIN_STOCKS = 5
    # 最低購入金額。購入額がこの金額より少ない場合は、超えるまでリスト内の銘柄を購入する
    BOUGHT_MIN = 500000
    # 予算。この金額以上は購入しない
    BOUGHT_MAX = 1000000

    # 入力データ準備
    @classmethod
    def prepare_data(cls, data_dir: str):
        # バックテストへの入力用データを作成します

        UNIVERSE_COL = "universe_comp2"
        CODE_COL = "Local Code"

        # ユニバース取得
        # 銘柄情報読み込み
        df_stock_list = pd.read_csv(f"{data_dir}/stock_list.csv.gz")
        # 問題2のユニバース取得
        codes = df_stock_list.loc[
            df_stock_list.loc[:, UNIVERSE_COL] == True, CODE_COL
        ].values

        # 価格情報読み込み
        df_price = pd.read_csv(f"{data_dir}/stock_price.csv.gz")
        # インデックス設定
        df_price.set_index("EndOfDayQuote Date", inplace=True)
        # インデックスを日付型に変更
        df_price.index = pd.to_datetime(df_price.index)
        # ユニバースに絞り込む
        df_price = df_price.loc[df_price.loc[:, CODE_COL].isin(codes)]

        return codes, df_price

    # 提出用データ読み込み
    @classmethod
    def load_submit(cls, file_path: str) -> pd.DataFrame:
        # データ読み込み
        df = pd.read_csv(file_path)

        # カラム存在検証
        if cls.DATE not in df.columns:
            raise ValueError(f"{cls.DATE} is missing in columns: {df.columns}")
        if cls.CODE not in df.columns:
            raise ValueError(f"{cls.CODE} is missing in columns: {df.columns}")
        if cls.BUDGET not in df.columns:
            raise ValueError(f"{cls.BUDGET} is missing in columns: {df.columns}")

        # インデックス設定
        df.set_index(cls.DATE, inplace=True)
        df.index = pd.to_datetime(df.index, format=cls.DATE_FORMAT)

        # データ型を設定
        df.loc[:, cls.CODE] = df.loc[:, cls.CODE].astype(np.int64)
        df.loc[:, cls.BUDGET] = df.loc[:, cls.BUDGET].astype(np.int64)

        # レコード順を保存
        df.loc[:, cls.BUYING_ORDER] = np.arange(df.shape[0])

        return df.loc[:, [cls.CODE, cls.BUDGET, cls.BUYING_ORDER]]

    # バックテスト実行関数
    @classmethod
    def calc_trades(cls, df_t, df_price):
        """
        入力:
            df_t (DataFrame): １週間分のポートフォリオ
            df_price (DataFrame): 株価情報
        出力:
            d_sum (dict[str, float]): 購入総額及びリターンの基準となる評価額
            buff_holiday  (list[int]): 祝日などで株価の存在しなかった曜日 (0: 月曜日、5: 金曜日)
            df_t (DataFrame): 銘柄の売り買い履歴
        """
        # 購入順に並び替え
        df_t = df_t.sort_values(cls.BUYING_ORDER)

        # 対象日付を取得
        s = df_t.index[0]

        # 予測対象の銘柄コードのみに絞り込み
        df_t = df_t.loc[df_t.loc[:, cls.CODE].isin(df_price.loc[:, cls.CODE].unique())]

        # 月曜日以外の場合はエラー
        if s.dayofweek != 0:
            raise ValueError(f"date:{s} is not Monday")

        # indexをユニーク値に変更
        df_t = df_t.reset_index()

        # 銘柄コードを取得
        stock_codes = df_t.loc[:, cls.CODE].unique()

        # 5銘柄以上選択されているかを確認
        if len(stock_codes) < 5:
            # 4銘柄以下の場合エラー
            raise ValueError(f"num of stocks should be more than or equal to 5. {df_t}")

        #  株価情報を対象日付で絞り込む
        df_price = df_price.sort_index().loc[s : s + pd.Timedelta("4D")]
        # 銘柄コードと日付で絞り込んだ株価を用意しておく
        dfp = {}
        # 銘柄コード毎に処理
        for stock_code in stock_codes:
            # 曜日ごとに処理
            for t in range(5):
                # 各曜日の日付を取得
                t_index = s + pd.Timedelta(f"{t}D")
                # 各銘柄、各曜日ごとの価格情報を保存
                dfp[(stock_code, t_index)] = df_price.loc[
                    (df_price.index == t_index)
                    & (df_price.loc[:, cls.CODE] == stock_code)
                ]

        # カラムを追加
        for col in ["entry", "day_1", "day_2", "day_3", "day_4", "day_5", "bought"]:
            df_t.loc[:, col] = 0.0
        df_t.loc[:, "actual"] = 0

        # 購入金額および日々の評価額保存用
        d_sum = {
            "bought_sum": 0,
            "day_1": 0,
            "day_2": 0,
            "day_3": 0,
            "day_4": 0,
            "day_5": 0,
        }

        # 祝日保存用
        buff_holiday = []
        # 購入ステータスフラグ
        buying = True
        salvation = False

        # メインループ: 銘柄を購入して日次の評価額を計算する
        # 1日毎に処理
        for offset in range(5):
            # 対象日付を取得
            t_index = s + pd.Timedelta(f"{offset}D")
            ##########
            # 祝日判定
            ##########
            if df_price.loc[(df_price.index == t_index)].shape[0] == 0:
                # 前日の評価額を引き継ぎ
                if offset == 0:
                    # 現金をそのままスライド
                    d_sum[f"day_{offset + 1}"] = cls.BOUGHT_MAX
                else:
                    d_sum[f"day_{offset + 1}"] = d_sum[f"day_{offset}"]
                # 祝日を保存
                buff_holiday.append(offset)
                continue
            ##########
            # 最初の営業日
            ##########
            if buying:
                # 購入フラグを設定
                buying = False
                # 処理回数カウント用
                iter_count = 0
                # 最低購入額を満たすまで実行
                while df_t["bought"].sum() < cls.BOUGHT_MIN:
                    # 処理回数を記録
                    iter_count += 1
                    # 処理回数が5万回を超える場合はエラーとする
                    if iter_count > 50000:
                        raise ValueError(f"[FILL] cannot not buy stocks: {df_t} ")

                    # レコード毎に処理
                    for i in df_t.index:
                        # 対象の銘柄コード
                        stock_code = df_t.loc[i, cls.CODE]
                        # 価格情報
                        df_p = dfp[(stock_code, t_index)]
                        # 価格情報がなかった場合 (上場前や廃止が考えられる)
                        if df_p.shape[0] == 0:
                            continue
                        # 値段がついていない場合
                        if df_p.iloc[0]["EndOfDayQuote Open"] == 0.0:
                            # 終値
                            df_t.loc[i, f"day_{offset + 1}"] = df_p.iloc[0][
                                "EndOfDayQuote ExchangeOfficialClose"
                            ]
                            continue
                        if salvation:
                            # 購入数量指定 (1固定)
                            buying_qty = 1
                        else:
                            # 予算を取得
                            buying_budget = min(
                                # レコードの予算額
                                df_t.loc[i, cls.BUDGET],
                                # 全体の予算額
                                cls.BOUGHT_MAX - df_t["bought"].sum(),
                            )
                            # 指定された金額での購入数量算出
                            buying_qty = int(
                                buying_budget
                                // (
                                    df_p.iloc[0]["EndOfDayQuote Open"]
                                    * df_p.iloc[0][
                                        "EndOfDayQuote CumulativeAdjustmentFactor"
                                    ]
                                )
                            )
                        # 累積調整係数を使用して購入数量を調整後数量に変更する
                        buying_qty *= df_p.iloc[0][
                            "EndOfDayQuote CumulativeAdjustmentFactor"
                        ]
                        # 購入金額算出
                        buying_amount = df_p.iloc[0]["EndOfDayQuote Open"] * buying_qty
                        # 終値基準価格算出
                        close_amount = (
                            df_p.iloc[0]["EndOfDayQuote ExchangeOfficialClose"]
                            * buying_qty
                        )
                        # 購入した場合に合計予算を超えないことを確認
                        tmp_bought_sum = df_t["bought"].sum() + buying_amount
                        if tmp_bought_sum > cls.BOUGHT_MAX:
                            # 予算を超える場合は購入しない
                            continue
                        # 購入額
                        d_sum["bought_sum"] += buying_amount
                        # # 購入額
                        # d_sum[f"day_{offset + 1}_open"] += buying_amount
                        # # 終値
                        d_sum[f"day_{offset + 1}"] += close_amount
                        df_t.loc[i, "bought"] += buying_amount
                        # 購入履歴
                        df_t.loc[i, "actual"] += buying_qty
                        # 購入価格
                        df_t.loc[i, "entry"] = df_p.iloc[0]["EndOfDayQuote Open"]
                        # 終値
                        df_t.loc[i, f"day_{offset + 1}"] = df_p.iloc[0][
                            "EndOfDayQuote ExchangeOfficialClose"
                        ]

                        # 購入金額が閾値以上となったら購入停止
                        if salvation and df_t["bought"].sum() >= cls.BOUGHT_MIN:
                            break
                    # 1株ずつ購入するように設定
                    salvation = True

                # 購入銘柄数が5未満の場合はエラーにする
                if (
                    len(df_t.loc[df_t.loc[:, "actual"] > 0, "Local Code"].unique())
                    < cls.MIN_STOCKS
                ):
                    raise ValueError(
                        f"Number of bought stocks is less than 5: {t_index}"
                    )
                # 最初の営業日の資産計上
                d_sum[f"day_{offset + 1}"] += cls.BOUGHT_MAX - d_sum["bought_sum"]
                # 最初の営業日終了
                continue

            # 2日目から5日目
            # for stock_code in buff_trans.keys():
            for i in df_t.index:
                stock_code = df_t.loc[i, cls.CODE]
                df_p = dfp[(stock_code, t_index)]
                if df_p.shape[0] == 0:
                    # レコードが見つからない場合は上場廃止が考えられる
                    # 救済措置としてExchangeOfficialCloseをffillして対応する
                    print(f"DELISTED: code: {stock_code}, no data: {t_index}")
                    # 終値ベースでの保有額
                    d_sum[f"day_{offset + 1}"] += (
                        df_t.loc[i, f"day_{offset}"] * df_t.loc[i, "actual"]
                    )
                    # 終値
                    df_t.loc[i, f"day_{offset + 1}"] = df_t.loc[i, f"day_{offset}"]
                    continue
                # 終値ベースでの保有額
                d_sum[f"day_{offset + 1}"] += (
                    df_p.iloc[0]["EndOfDayQuote ExchangeOfficialClose"]
                    * df_t.loc[i, "actual"]
                )
                # 終値
                df_t.loc[i, f"day_{offset + 1}"] = df_p.iloc[0][
                    "EndOfDayQuote ExchangeOfficialClose"
                ]
            # 営業日の資産計上
            d_sum[f"day_{offset + 1}"] += cls.BOUGHT_MAX - d_sum["bought_sum"]

        # 購入銘柄数が閾値未満の場合はエラー
        if (len(buff_holiday) != 5) and (
            len(df_t.loc[df_t.loc[:, "actual"] > 0, "Local Code"].unique())
            < cls.MIN_STOCKS
        ):
            raise ValueError(f"bought_count is less than {cls.MIN_STOCKS}")

        # 購入金額がしきい値未満の場合はエラー
        bought_sum = df_t["bought"].sum()
        if (len(buff_holiday) != 5) and (bought_sum < cls.BOUGHT_MIN):
            raise ValueError(f"bought_sum:{bought_sum} is less than {cls.BOUGHT_MIN}")

        return (
            d_sum,
            buff_holiday,
            df_t,
        )

    @classmethod
    def run(cls, df_submit, stock_codes, df_price):
        """
        入力:
            df_submit (DataFrame): ポートフォリオ
            stock_codes (list): ユニバース
            df_price (DataFrame): 株価情報
        出力:
            df_return (DataFrame): ポートフォリオのリターン
            df_transaction (DataFrame): 銘柄の売り買い履歴
        """
        MONDAY = 0

        buff_return = []
        buff_stocks = []

        # Nanが含まれていないことを確認
        if df_submit.isnull().values.any():
            raise ValueError("This DataFrame includes Nan values")

        # 入力データ確認
        # 1:買い注文の銘柄が予測対象かをチェック
        # 予測対象銘柄のみに絞り込む
        df_submit = df_submit.loc[df_submit.loc[:, cls.CODE].isin(stock_codes)]

        # 2:月曜日の日付のみを評価対象とする
        # 月曜日の日付のみに絞り込む
        df_submit = df_submit.loc[df_submit.index.dayofweek == MONDAY]

        # 購入金額が1未満のレコードを除外する
        df_submit = df_submit.loc[df_submit.loc[:, cls.BUDGET] >= 1]

        # 提出データを１週間毎にリターンを計算する
        uniq_dates = df_submit.index.unique()
        for s in tqdm(uniq_dates):
            # 対象日付データに絞り込み
            df_t = df_submit.loc[df_submit.index == s]
            # 銘柄コードを取得
            stock_codes = sorted(df_t[cls.CODE].unique())
            # 金曜日日付を取得
            friday = s + pd.Timedelta("4D")
            # 金曜日基準の株価を取得
            df_price_on_friday = Backtest.adjust_price(
                stock_codes,
                df_price.loc[(df_price.index >= s) & (df_price.index <= friday)],
            )

            # バックテスト実行
            (
                d_sum,
                buff_holiday,
                df_t,
            ) = Backtest.calc_trades(df_t, df_price_on_friday)

            # 現金を計算
            cash = cls.BOUGHT_MAX - d_sum["bought_sum"]

            # 週の開始日付(月曜日)
            d_sum["date"] = s
            # 祝日
            d_sum["holiday"] = buff_holiday
            # 購入総額
            d_sum["bought"] = d_sum["bought_sum"]
            # 現金
            d_sum["cash"] = cash
            # ポートフォリオの週次PLを計算
            d_sum["week_pl"] = d_sum["day_5"] - cls.BOUGHT_MAX
            # ポートフォリオの週次リターンを計算
            d_sum["week_return"] = ((d_sum["day_5"] / cls.BOUGHT_MAX) - 1) * 100

            for i in [1, 2, 3, 4, 5]:
                if i == 1:
                    d_sum[f"day_{i}_return"] = (
                        (d_sum[f"day_{i}"] / cls.BOUGHT_MAX) - 1
                    ) * 100
                    d_sum[f"day_{i}_pl"] = d_sum[f"day_{i}"] - cls.BOUGHT_MAX
                else:
                    # ポートフォリオの日次リターンを計算
                    d_sum[f"day_{i}_return"] = (
                        ((d_sum[f"day_{i}"]) / (d_sum[f"day_{i-1}"])) - 1
                    ) * 100
                    # ポートフォリオの日次PLを計算
                    d_sum[f"day_{i}_pl"] = d_sum[f"day_{i}"] - d_sum[f"day_{i-1}"]

            # 祝日を計算式から除外する
            n = 5 - len(buff_holiday)
            buff_exp = 0.0
            buff_std = 0.0
            # 日次の期待リターンを計算
            for i in range(5):
                if i not in buff_holiday:
                    buff_exp += d_sum[f"day_{i+1}_return"]
            d_sum["exp"] = buff_exp / n if n != 0 else 0.0
            # 日次リターンの標準偏差を計算
            for i in range(5):
                if i not in buff_holiday:
                    buff_std += (d_sum[f"day_{i+1}_return"] - d_sum["exp"]) * (
                        d_sum[f"day_{i+1}_return"] - d_sum["exp"]
                    )
            d_sum["std"] = np.sqrt(buff_std / (n - 1)) if n != 1 else 0.0
            # シャープレシオ を計算
            d_sum["sharp"] = d_sum["exp"] / d_sum["std"] if d_sum["std"] != 0 else 0.0

            del d_sum["bought_sum"]

            # 結果を保存
            buff_return.append(d_sum)

            # レコードごとの履歴を保存
            buff_stocks.append(df_t)

        df_return = pd.DataFrame(buff_return, index=range(len(buff_return)))
        df_stocks = pd.concat(buff_stocks)
        return df_return, df_stocks

    @classmethod
    def adjust_price(cls, stock_codes: list, df_price: pd.DataFrame) -> pd.DataFrame:
        adjust_target_columns_multiply = [
            "EndOfDayQuote Open",
            # "EndOfDayQuote High",
            # "EndOfDayQuote Low",
            # "EndOfDayQuote Close",
            "EndOfDayQuote ExchangeOfficialClose",
            # "EndOfDayQuote PreviousClose",
            # "EndOfDayQuote PreviousExchangeOfficialClose",
            # "EndOfDayQuote ChangeFromPreviousClose",
            # "EndOfDayQuote VWAP",
        ]
        adjust_target_columns_divide = [
            # "EndOfDayQuote Volume",
            "EndOfDayQuote CumulativeAdjustmentFactor",
        ]

        df_price = df_price.sort_values(["Local Code", "EndOfDayQuote Date"])

        for stock_code in stock_codes:
            filter_stock = df_price["Local Code"] == stock_code

            # get latest adjustment factor
            latest_adjustment_factor = df_price.loc[filter_stock].iloc[-1][
                "EndOfDayQuote CumulativeAdjustmentFactor"
            ]

            # adjust values
            for col in adjust_target_columns_multiply:
                df_price.loc[filter_stock, col] = (
                    df_price.loc[filter_stock, col] * latest_adjustment_factor
                )
            for col in adjust_target_columns_divide:
                df_price.loc[filter_stock, col] = (
                    df_price.loc[filter_stock, col] / latest_adjustment_factor
                )

        return df_price
