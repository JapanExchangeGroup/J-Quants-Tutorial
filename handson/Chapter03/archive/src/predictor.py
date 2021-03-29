# -*- coding: utf-8 -*-
import io

import pandas as pd


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path="../model"):
        return True

    @classmethod
    def predict(
        cls, inputs, start_dt=pd.Timestamp("2021-02-01"), strategy_id="reversal"
    ):
        ####
        # データセットを読み込みます
        ####
        # 銘柄情報読み込み
        df_stock_list = pd.read_csv(inputs["stock_list"])
        # 問題2のユニバース (投資対象銘柄群) 取得
        codes = df_stock_list.loc[
            df_stock_list.loc[:, "universe_comp2"] == True, "Local Code"
        ].unique()

        # 価格情報読み込み、インデックス作成
        df_price = pd.read_csv(inputs["stock_price"]).set_index("EndOfDayQuote Date")
        # 日付型に変換
        df_price.index = pd.to_datetime(df_price.index, format="%Y-%m-%d")

        if "purchase_date" in inputs.keys():
            # ランタイム環境では指定された投資対象日付を使用します
            # purchase_dateを読み込み
            df_purchase_date = pd.read_csv(inputs["purchase_date"])
            # purchase_dateの最も古い日付を設定
            start_dt = pd.Timestamp(
                df_purchase_date.sort_values("Purchase Date").iloc[0, 0]
            )

        # 投資対象日の前週金曜日時点で予測を出力するため、予測出力用の日付を設定します。
        pred_start_dt = pd.Timestamp(start_dt) - pd.Timedelta("3D")
        # 特徴量の生成に必要な日数をバッファとして設定
        n = 30
        # データ絞り込み日付設定
        data_start_dt = pred_start_dt - pd.offsets.BDay(n)
        # 日付で絞り込み
        filter_date = df_price.index >= data_start_dt
        # 銘柄をユニバースで絞り込み
        filter_universe = df_price.loc[:, "Local Code"].isin(codes)
        # 絞り込み実施
        df_price = df_price.loc[filter_date & filter_universe]

        ####
        # シンプルな特徴量を作成します
        ####
        # groupby を使用して処理するために並び替え
        df_price.sort_values(["Local Code", "EndOfDayQuote Date"], inplace=True)
        # 銘柄毎にグループにします。
        grouped_price = df_price.groupby("Local Code")[
            "EndOfDayQuote ExchangeOfficialClose"
        ]
        # 銘柄毎に20営業日の変化率を作成してから、金曜日に必ずデータが存在するようにリサンプルしてフィルします
        df_feature = grouped_price.apply(
            lambda x: x.pct_change(20).resample("B").ffill().dropna()
        ).to_frame()

        # 上記が比較的時間のかかる処理なので、処理済みデータを残しておきます。
        df_work = df_feature  # copyはランタイム実行時には不要なので削除しています

        # インデックスが銘柄コードと日付になっているため、日付のみに変更します。
        df_work = df_work.reset_index(level=[0])
        # カラム名を変更します
        df_work.rename(
            columns={"EndOfDayQuote ExchangeOfficialClose": "pct_change"},
            inplace=True,
        )
        # データをpred_start_dt以降の日付に絞り込みます
        df_work = df_work.loc[df_work.index >= pred_start_dt]

        ####
        # ポートフォリオを組成します
        ####
        # 金曜日のデータのみに絞り込みます
        df_work = df_work.loc[df_work.index.dayofweek == 4]

        # 日付毎に処理するためグループ化します
        grouped_work = df_work.groupby("EndOfDayQuote Date", as_index=False)

        # 選択する銘柄数を指定します
        number_of_portfolio_stocks = 25

        # ポートフォリオの組成方法を戦略に応じて調整します
        strategies = {
            # リターン・リバーサル戦略
            "reversal": {"asc": True},
            # トレンドフォロー戦略
            "trend": {"asc": False},
        }

        # 戦略に応じたポートフォリオを保存します
        df_portfolios = {}

        # strategy_id が設定されていない場合は全ての戦略のポートフォリオを作成します
        if "strategy_id" not in locals():
            strategy_id = None

        for i in [strategy_id] if strategy_id is not None else strategies.keys():
            #  日付毎に戦略に応じた上位25銘柄を選択します。
            df_portfolios[i] = grouped_work.apply(
                lambda x: x.sort_values(
                    "pct_change", ascending=strategies[i]["asc"]
                ).head(number_of_portfolio_stocks)
            )

        # 銘柄ごとの購入金額を指定
        budget = 50000
        # 戦略毎に処理
        for i in df_portfolios.keys():
            # 購入株式数を設定
            df_portfolios[i].loc[:, "budget"] = budget
            # インデックスを日付のみにします
            df_portfolios[i].reset_index(level=[0], inplace=True)
            # 金曜日から月曜日日付に変更
            df_portfolios[i].index = df_portfolios[i].index + pd.Timedelta("3D")

        ####
        # 出力を調整します
        ####
        # 戦略毎に処理
        for i in df_portfolios.keys():
            # インデックス名を設定
            df_portfolios[i].index.name = "date"
            # 出力するカラムを絞り込みます
            df_portfolios[i] = df_portfolios[i].loc[:, ["Local Code", "budget"]]

        # 出力保存用
        outputs = {}
        # 戦略毎に処理
        for i in df_portfolios.keys():
            # 出力します
            out = io.StringIO()
            # CSV形式で出力
            df_portfolios[i].to_csv(out, header=True)
            # 出力を保存
            outputs[i] = out.getvalue()

        return outputs[strategy_id]
