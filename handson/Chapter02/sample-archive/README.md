# シンプルなモデルの作成手順

## 方針説明

本文章は「ファンダメンタルズ分析チャレンジ」においてシンプルなモデルを作成することを目的としています。

本書を読みながら一連のステップを完了するための必要時間はおおよそ20分です。

以下のステップで進めます。

1. 前提条件
1. コンペティションページからデータの取得
1. Google Drive にデータを配置
1. Google Colaboratory で新規ノートブックの作成
1. Google Drive をマウント
1. データの読み込み
1. 特徴量の作成
1. モデル出力を調整
1. パッケージ化

## 前提条件

本書は以下の事項を前提条件として記載されています。

1. 「ファンダメンタルズ分析チャレンジ」にご参加いただいていること
1. Google アカウントを保有されていること
1. Google Colaboratory を使用できること
1. インターネットに接続されていること
1. Pythonについて基礎的な知識をお持ちであること
1. Pandasについて基礎的な知識をお持ちであること

## コンペティションページのデータタブより以下のデータをダウンロード

データタブ: https://signate.jp/competitions/423/data

- stock_price (stock_price.csv.gz)

## Google Drive にデータを配置

Google Drive: https://drive.google.com/

1. My Drive 配下に `JPX_competition` フォルダを作成します。
1. `JPX_competition` フォルダにダウンロードした `stock_price.csv.gz` ファイルを配置します。

## Google Colaboratory で新規ノートブックの作成

1. Google Colaboratory を開きます。https://colab.research.google.com/
1. メニューバーの「ファイル」から「ノートブックを新規作成」を選択します。
1. ノートブック名を「jpx-comp1-simple-model.ipynb」とします。
1. 画面右上にある「接続」を押下してランタイムに接続します。
1. 必要なモジュールを読み込んでおきます。

    ```python
    import os
    import io

    import pandas as pd
    ```

## Google Drive をマウント

1. 先程配置した `stock_price` を読み込むために Google Drive をマウントします。
1. 画面左端サイドバーの一番下に配置されているフォルダのアイコンを押下して開いたペインで、Google Driveのアイコン(「ドライブをマウント」)を押下します。
1. ダイアログで「GOOGLE ドライブに接続」を押下します。
1. これで `/content/drive/MyDrive/JPX_competition` というパスでアクセスできます。

    ```python
    dataset_dir = "/content/drive/MyDrive/JPX_competition"
    os.path.isdir(dataset_dir)
    ```

## データの読み込み

1. データを読み込む際にランタイム環境と同一方法で読み込むために、inputs変数を作ります。

    ```python
    inputs = {"stock_price": f"{dataset_dir}/stock_price.csv.gz"}
    ```

1. データを読み込みます。

    ```python
    df = pd.read_csv(inputs["stock_price"])
    df.head(1)
    ```

1. インデックスを調整します。

    ```python
    df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, "EndOfDayQuote Date"])
    df.set_index("datetime", inplace=True)
    df.head(1)
    ```

## 特徴量の作成

1. 特徴量を作成します。今回は過去20営業日のボラティリティが最高値・最安値と相関している、すなわち、決算発表があった日から過去20営業日間のヒストリカル・ボラティリティが大きければ、決算発表から20営業日後の間の最高値・最安値への変化率も大きくなるという仮説を基に、ヒストリカル・ボラティリティを採用しています。

```python
feats = (
    df[["EndOfDayQuote ExchangeOfficialClose", "Local Code"]]
    .groupby("Local Code")
    .pct_change()
    .rolling(20)
    .std()
    .values
)
feats[:21]
```

## モデル出力を調整

1. ランタイム環境でのモデル出力要件に合わせてデータフレームを調整します。

    ```python
    # 結果を以下のcsv形式で出力する
    # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
    # ２列目:label_high_20　終値→最高値への変化率
    # ３列目:label_low_20　終値→最安値への変化率
    # headerはなし、２列３列はfloat64
    ```

1. code列を作成して出力形式の１列目と一致させる

    ```python
    df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[
        :, "Local Code"
    ].astype(str)
    df.head(1).T
    ```

1. ヒストリカル・ボラティリティを予測値として２列３列目に設定します。これは、評価方法が順位相関であるため、大小関係を一致させれば予測値のとりうる範囲は関係ないためこの方法を使用しています。

    ```python
    df.loc[:, "label_high_20"] = feats
    df.loc[:, "label_low_20"] = feats
    ```

1. CSV形式で出力する列を指定します。

    ```python
    # 出力対象列を設定
    output_columns = ["code", "label_high_20", "label_low_20"]
    ```

1. 結果を出力します。Nanが含まれないように `dropna()` しています。

   ```python
    out = io.StringIO()
    df[output_columns].dropna().to_csv(out, header=False, index=False)

    out.getvalue()[-100:]
    ```

## パッケージ化

1. 提出用にこれまでのコードから必要部分をScoringServiceとして以下のようにコピーします。

    ```python
    import io

    import pandas as pd

    class ScoringService(object):
        @classmethod
        def get_model(cls, model_path="../model"):
            return True

        @classmethod
        def predict(cls, inputs):
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
            df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[
            :, "Local Code"
            ].astype(str)
            df.loc[:, "label_high_20"] = feats
            df.loc[:, "label_low_20"] = feats
            output_columns = ["code", "label_high_20", "label_low_20"]
            out = io.StringIO()
            df[output_columns].dropna().to_csv(out, header=False, index=False)

            return out.getvalue()
    ```

1. get_model をテストします。

    `assert` ステートメントは `True` 以外の値のときに `AssertionError` 例外を発生させるため、一般的にはデバッグやテスト用途で使用されます。今回はメソッドの戻り値が期待通りの値かどうかを検証するために使用しています。

    ```python
    assert ScoringService.get_model()
    ```

1. predict をテストします。

    ```python
    assert out.getvalue() == ScoringService.predict(inputs)
    ```

1. パッケージの構造を再確認します。

    https://signate.jp/features/runtime/detail
    以下のディレクトリ構造であること。

    ```directory
    .
    ├── model              必須 学習済モデルを置くディレクトリ
    │   └── ...
    ├── src                必須 Python のプログラムを置くディレクトリ
    │   ├── predictor.py       必須 最初のプログラムが呼び出すファイル
    │   └── ...              その他のファイル (ディレクトリ作成可能)
    └── requirements.txt   任意
    ```

1. ScoringServiceクラスをpredictor.pyファイルに保存します。

1. modelディレクトリをzipファイルに含めるためにダミーファイルを作成します。

    ```bash
    touch model/dummy.txt
    ```

1. zipで圧縮します。

    ```bash
    $ ls
    model src
    $ zip -v submit.zip src/predictor.py model/dummy.txt
    ```
