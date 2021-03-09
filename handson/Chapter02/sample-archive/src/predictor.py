# -*- coding: utf-8 -*-
import io

import pandas as pd


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path="../model"):
        return True

    @classmethod
    def predict(cls, inputs):

        # 特徴量を作成
        df = pd.read_csv(inputs["stock_price"])
        df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, "EndOfDayQuote Date"])
        df.set_index("datetime", inplace=True)

        feats = (
            df[["EndOfDayQuote ExchangeOfficialClose", "Local Code"]]
            .groupby("Local Code")
            .pct_change()
            .rolling(20)
            .std()
            .values
        )

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[
            :, "Local Code"
        ].astype(str)

        # ボラティリティを予測値として使用
        # 評価方法が順位相関であるため、大小関係を一致させれば
        # 予測値のとりうる範囲は関係ないためこの方法を使用可能です。
        df.loc[:, "label_high_20"] = feats
        df.loc[:, "label_low_20"] = feats
        # 出力対象列を設定
        output_columns = ["code", "label_high_20", "label_low_20"]

        out = io.StringIO()
        df[output_columns].dropna().to_csv(out, header=False, index=False)

        return out.getvalue()
