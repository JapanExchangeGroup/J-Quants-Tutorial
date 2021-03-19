# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from module import SentimentGenerator
from scipy.stats import zscore
from tqdm.auto import tqdm


class ScoringService(object):
    # テスト期間開始日
    TEST_START = "2021-02-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    # センチメントの分布をこの変数に読み込む
    df_sentiment_dist = None

    @classmethod
    def get_dataset(cls, inputs, load_data):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            # 必要なデータのみ読み込みます
            if k not in load_data:
                continue
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["universe_comp2"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_for_predict2(cls, dfs, code, fin_columns, start_dt=TEST_START):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            fin_columns(list[str]): list of columns
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_fin_priceデータを読み込み
        stock_fin_price = dfs["stock_fin_price"]

        # 特定の銘柄コードのデータに絞る
        feats = stock_fin_price[stock_fin_price["Local Code"] == code]
        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        # 指定されたカラムおよびExchangeOfficialCloseに絞り込み
        feats = feats.loc[
            :, fin_columns + ["EndOfDayQuote ExchangeOfficialClose"]
        ].copy()
        # 欠損値処理
        feats = feats.fillna(0)

        # 終値の20営業日リターン
        feats["return_1month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(20)
        # 終値の40営業日リターン
        feats["return_2month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(60)
        # 終値の20営業日ボラティリティ
        feats["volatility_1month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(20)
            .std()
        )
        # 終値の40営業日ボラティリティ
        feats["volatility_2month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(40)
            .std()
        )
        # 終値の60営業日ボラティリティ
        feats["volatility_3month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(60)
            .std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(20).mean()
        )
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(40).mean()
        )
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(60).mean()
        )
        # 欠損値処理
        feats = feats.fillna(0)
        # 元データのカラムを削除
        feats = feats.drop(["EndOfDayQuote ExchangeOfficialClose"], axis=1)

        # 1B resample + ffill で金曜日に必ずレコードが存在するようにする
        feats = feats.resample("B").ffill()
        # 特徴量を金曜日日付のみに絞り込む
        FRIDAY = 4
        feats = feats.loc[feats.index.dayofweek == FRIDAY]

        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], 0)

        # 銘柄コードを設定
        feats["code"] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X, column_group="fundamental+technical"):
        # 特徴量グループを定義
        # ファンダメンタル
        fundamental_cols = dfs["stock_fin"].select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[
            fundamental_cols != "Result_Dividend DividendPayableDate"
        ]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        # 価格変化率
        returns_cols = [x for x in train_X.columns if "return" in x]
        # テクニカル
        technical_cols = [
            x for x in train_X.columns if (x not in fundamental_cols) and (x != "code")
        ]
        columns = {
            "fundamental_only": fundamental_cols,
            "return_only": returns_cols,
            "technical_only": technical_cols,
            "fundamental+technical": list(fundamental_cols) + list(technical_cols),
        }
        return columns[column_group]

    @classmethod
    def transform_yearweek_to_monday(cls, year, week):
        """
        ニュースから抽出した特徴量データのindexは (year, week) なので、
        (year, week) => YYYY-MM-DD 形式(月曜日) に変換します。
        """
        for s in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            if s.week == week:
                # to return Monday of the first week of the year
                # e.g. "2020-01-01" => "2019-12-30"
                return s - pd.Timedelta(f"{s.dayofweek}D")

    @classmethod
    def load_sentiments(cls, path=None):
        DIST_END_DT = "2020-09-25"

        print(f"[+] load prepared sentiment: {path}")

        # 事前に出力したセンチメントの分布を読み込み
        df_sentiments = pd.read_pickle(path)

        # indexを日付型に変換します変換します。
        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        # indexを設定します
        df_sentiments.set_index("index", inplace=True)
        # カラム名を変更します
        df_sentiments.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        # 分布として使用するデータの範囲に絞り込みます
        df_sentiments = df_sentiments.loc[:DIST_END_DT]

        # 金曜日日付に変更します
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")

        return df_sentiments

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            m = os.path.join(model_path, f"my_model_{label}.pkl")
            with open(m, "rb") as f:
                # pickle形式で保存されているモデルを読み込み
                cls.models[label] = pickle.load(f)

        # SentimentGeneratorクラスの初期設定を実施
        SentimentGenerator.initialize(model_path)

        # 事前に計算済みのセンチメントを分布として使用するために読み込みます
        cls.df_sentiment_dist = cls.load_sentiments(
            f"{model_path}/headline_features/LSTM_sentiment.pkl"
        )

        return True

    @classmethod
    def get_exclude(
        cls,
        df_tdnet,  # tdnetのデータ
        start_dt=None,  # データ取得対象の開始日、Noneの場合は制限なし
        end_dt=None,  # データ取得対象の終了日、Noneの場合は制限なし
        lookback=7,  # 除外考慮期間 (days)
        target_day_of_week=4,  # 起点となる曜日
    ):
        # 特別損失のレコードを取得
        special_loss = df_tdnet[df_tdnet["disclosureItems"].str.contains('201"')].copy()
        # 日付型を調整
        special_loss["date"] = pd.to_datetime(special_loss["disclosedDate"])
        # 処理対象開始日が設定されていない場合はデータの最初の日付を取得
        if start_dt is None:
            start_dt = special_loss["date"].iloc[0]
        # 処理対象終了日が設定されていない場合はデータの最後の日付を取得
        if end_dt is None:
            end_dt = special_loss["date"].iloc[-1]
        #  処理対象日で絞り込み
        special_loss = special_loss[
            (start_dt <= special_loss["date"]) & (special_loss["date"] <= end_dt)
        ]
        # 出力用にカラムを調整
        res = special_loss[["code", "disclosedDate", "date"]].copy()
        # 銘柄コードを4桁にする
        res["code"] = res["code"].astype(str).str[:-1]
        # 予測の基準となる金曜日の日付にするために調整
        res["remain"] = (target_day_of_week - res["date"].dt.dayofweek) % 7
        res["start_dt"] = res["date"] + pd.to_timedelta(res["remain"], unit="d")
        res["end_dt"] = res["start_dt"] + pd.Timedelta(days=lookback)
        # 出力するカラムを指定
        columns = ["code", "date", "start_dt", "end_dt"]
        return res[columns].reset_index(drop=True)

    @classmethod
    def strategy(cls, strategy_id, df, df_tdnet):
        df = df.copy()
        # 銘柄選択方法選択
        if strategy_id in [1, 4]:
            # 最高値モデル +　最安値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"] + df.loc[:, "label_low_20"]
        elif strategy_id in [2, 5]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"]
        elif strategy_id in [3, 6]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_low_20"]
        else:
            raise ValueError("no strategy_id selected")

        # 特別損失を除外する場合
        if strategy_id in [4, 5, 6]:
            # 特別損失が発生した銘柄一覧を取得
            df_exclude = cls.get_exclude(df_tdnet)
            # 除外用にユニークな列を作成します。
            df_exclude.loc[:, "date-code_lastweek"] = (
                df_exclude.loc[:, "start_dt"].dt.strftime("%Y-%m-%d-")
                + df_exclude.loc[:, "code"]
            )
            df_exclude.loc[:, "date-code_thisweek"] = (
                df_exclude.loc[:, "end_dt"].dt.strftime("%Y-%m-%d-")
                + df_exclude.loc[:, "code"]
            )
            #
            df.loc[:, "date-code_lastweek"] = (df.index - pd.Timedelta("7D")).strftime(
                "%Y-%m-%d-"
            ) + df.loc[:, "code"].astype(str)
            df.loc[:, "date-code_thisweek"] = df.index.strftime("%Y-%m-%d-") + df.loc[
                :, "code"
            ].astype(str)
            # 特別損失銘柄を除外
            df = df.loc[
                ~df.loc[:, "date-code_lastweek"].isin(
                    df_exclude.loc[:, "date-code_lastweek"]
                )
            ]
            df = df.loc[
                ~df.loc[:, "date-code_thisweek"].isin(
                    df_exclude.loc[:, "date-code_thisweek"]
                )
            ]

        # 予測出力を降順に並び替え
        df = df.sort_values("pred", ascending=False)
        # 予測出力の大きいものを取得
        df = df.groupby("datetime").head(30)

        return df

    @classmethod
    def predict(
        cls,
        inputs,
        labels=None,
        codes=None,
        start_dt=TEST_START,
        load_data=[
            "stock_list",
            "stock_fin",
            "stock_fin_price",
            "stock_price",
            "tdnet",
            "purchase_date",
        ],
        fin_columns=None,
        strategy_id=5,
    ):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify target purchase date
            load_data (list[str]): list of data to load
            fin_columns (list[str]): list of columns to use as features
            strategy_id (int): specify strategy to use
        Returns:
            str: Inference for the given input.
        """
        if fin_columns is None:
            fin_columns = [
                "Result_FinancialStatement FiscalYear",
                "Result_FinancialStatement NetSales",
                "Result_FinancialStatement OperatingIncome",
                "Result_FinancialStatement OrdinaryIncome",
                "Result_FinancialStatement NetIncome",
                "Result_FinancialStatement TotalAssets",
                "Result_FinancialStatement NetAssets",
                "Result_FinancialStatement CashFlowsFromOperatingActivities",
                "Result_FinancialStatement CashFlowsFromFinancingActivities",
                "Result_FinancialStatement CashFlowsFromInvestingActivities",
                "Forecast_FinancialStatement FiscalYear",
                "Forecast_FinancialStatement NetSales",
                "Forecast_FinancialStatement OperatingIncome",
                "Forecast_FinancialStatement OrdinaryIncome",
                "Forecast_FinancialStatement NetIncome",
                "Result_Dividend FiscalYear",
                "Result_Dividend QuarterlyDividendPerShare",
                "Result_Dividend AnnualDividendPerShare",
                "Forecast_Dividend FiscalYear",
                "Forecast_Dividend QuarterlyDividendPerShare",
                "Forecast_Dividend AnnualDividendPerShare",
            ]

        # データ読み込み
        if cls.dfs is None:
            print("[+] load data")
            cls.get_dataset(inputs, load_data)
            cls.get_codes(cls.dfs)

        # purchase_date が存在する場合は予測対象日を上書き
        if "purchase_date" in cls.dfs.keys():
            # purchase_dateの最も古い日付を設定
            start_dt = cls.dfs["purchase_date"].sort_values("Purchase Date").iloc[0, 0]

        # 日付型に変換
        start_dt = pd.Timestamp(start_dt)
        # 予測対象日の月曜日日付が指定されているため
        # 特徴量の抽出に使用する1週間前の日付に変換します
        start_dt -= pd.Timedelta("7D")
        # 文字列型に戻す
        start_dt = start_dt.strftime("%Y-%m-%d")

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        ###################
        # センチメント情報取得
        ###################
        # ニュース見出しデータへのパスを指定
        df_sentiments = cls.get_sentiment(inputs, start_dt=start_dt)
        #
        # 金曜日日付に変更
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")
        # 分布データを取り込み
        df_sentiments = pd.concat([cls.df_sentiment_dist, df_sentiments])

        # センチメントから現金比率を算出
        df_cash = cls.get_cash_ratio(df_sentiments)

        # 特徴量を作成
        print("[+] generate feature")
        buff = []
        for code in tqdm(codes):
            buff.append(
                cls.get_features_for_predict2(cls.dfs, code, fin_columns, start_dt)
            )
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # 1列目:date
        # 2列目:Local Code
        # 3列目:budget
        # headerあり、2列目3列目はint64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        print(df)
        # 購入金額を設定 (ここでは一律50000とする)
        df.loc[:, "budget"] = 50000
        print(df_cash)
        for s in df_cash.index:
            t = df_cash.loc[df_cash.index == s, "risk"][0]
            if t == 10:
                cash = 40000
            elif t == 20:
                cash = 30000
            elif t == 30:
                cash = 20000
            else:
                cash = 50000

            df.loc[df.index == s, "budget"] = cash

        # 特徴量カラムを指定
        feature_columns = ScoringService.get_feature_columns(cls.dfs, feats)

        # 目的変数毎に予測
        print("[+] predict")
        for label in tqdm(labels):
            # 予測実施
            df[label] = ScoringService.models[label].predict(feats[feature_columns])
            # 出力対象列に追加

        # 銘柄選択方法選択
        df = cls.strategy(strategy_id, df, cls.dfs["tdnet"])

        # 日付順に並び替え
        df.sort_index(kind="mergesort", inplace=True)
        # 月曜日日付に変更
        df.index = df.index + pd.Timedelta("3D")
        # 出力用に調整
        df.index.name = "date"
        df.rename(columns={"code": "Local Code"}, inplace=True)
        df.reset_index(inplace=True)

        # 出力対象列を定義
        output_columns = ["date", "Local Code", "budget"]

        out = io.StringIO()
        df.to_csv(out, header=True, index=False, columns=output_columns)

        return out.getvalue()

    @classmethod
    def get_sentiment(cls, inputs, start_dt="2020-12-31"):
        # ニュース見出しデータへのパスを指定
        article_path = inputs["nikkei_article"]
        target_feature_types = ["headline"]
        df_sentiments = SentimentGenerator.generate_lstm_features(
            article_path,
            start_dt=start_dt,
            target_feature_types=target_feature_types,
        )["headline_features"]

        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        df_sentiments.set_index("index", inplace=True)
        df_sentiments.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        return df_sentiments

    @classmethod
    def get_cash_ratio(cls, df_sentiment):
        """
        headline_m2_sentimentの値が低い時はリスクとみなし
        現金保有量を多くする。

        入力:
        センチメント (DataFrame): センチメント情報
        出力:
        現金比率 (Dataframe): 入力値に現金比率を追加したもの
        """
        # リスク値マッピング用の分布取得期間
        DIST_START_DT = "2020-06-29"
        DIST_END_DT = "2020-09-25"
        # リスク値を計算する期間
        USE_START_DT = "2020-10-02"

        # 出力用にコピー
        df_sentiment = df_sentiment.copy()
        # headline_m2_sentiment_0の値が高いほどポジティブなので符号反転させる
        sentiment_dist = sorted(
            df_sentiment.loc[
                DIST_START_DT:DIST_END_DT, "headline_m2_sentiment_0"
            ].values
            * -1
        )
        sentiment_use = (
            df_sentiment.loc[USE_START_DT:, "headline_m2_sentiment_0"].values * -1
        )

        # DIST_START_DT:DIST_END_DTの分布を使用してリスク判定する
        z = zscore(sentiment_dist)
        # 閾値を決定
        p = np.percentile(z, [25, 50, 75])
        # リスク値を計算する
        u = zscore(sentiment_use)
        # 分布から現金比率の割合を決定
        d = np.digitize(u, p)
        # 出力用に整形して 0, 10, 20, 30 のいずれか返すようにする
        df_sentiment.loc[USE_START_DT:, "risk"] = d * 10
        return df_sentiment
