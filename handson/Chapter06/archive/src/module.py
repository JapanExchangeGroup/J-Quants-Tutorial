import math
import os
import unicodedata
from glob import glob

import neologdn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset
from tqdm.auto import tqdm
from transformers import BertJapaneseTokenizer


class Dataset(_Dataset):
    def __init__(self, weekly_features, weekly_labels, max_sequence_length):
        # 共通する週のみを使うため、共通するindex情報を取得する
        mask_index = (
            weekly_features.index.get_level_values(0).unique() & weekly_labels.index
        )

        # 共通するindexのみのデータだけでreindexを行う。
        self.weekly_features = weekly_features[
            weekly_features.index.get_level_values(0).isin(mask_index)
        ]
        self.weekly_labels = weekly_labels.reindex(mask_index)

        # idからweekの情報を取得できるよう、id_to_weekをビルドする
        self.id_to_week = {
            id: week for id, week in enumerate(sorted(weekly_labels.index))
        }

        self.max_sequence_length = max_sequence_length

    def _shuffle_by_local_split(self, x, split_size=50):
        return torch.cat(
            [
                splitted[torch.randperm(splitted.size()[0])]
                for splitted in x.split(split_size, dim=0)
            ],
            dim=0,
        )

    def __len__(self):
        return len(self.weekly_labels)

    def __getitem__(self, id):
        # 付与されたidから週の情報を取得し、その週の情報から、特徴量とラベルを取得する。
        week = self.id_to_week[id]
        x = self.weekly_features.xs(week, axis=0, level=0)[-self.max_sequence_length :]
        y = self.weekly_labels[week]

        # pytorchでは、データをtorch.Tensorタイプとして扱うことが要求される。
        # 全体的な特徴量(ニュースの情報)の順序は維持しつつ、入力とする特徴量を数分割し、その分割の中でシャッフルを行う。
        x = self._shuffle_by_local_split(torch.tensor(x.values, dtype=torch.float))
        y = torch.tensor(y, dtype=torch.float)

        # max_sequence_lengthに最大のsequenceを合わせ、sequenceがmax_sequence_lengthに達しない場合は、前から0を埋め、sequenceを合わせる
        if x.size()[0] < self.max_sequence_length:
            x = F.pad(x, pad=(0, 0, self.max_sequence_length - x.size()[0], 0))

        return x, y


class FeatureCombiner(nn.Module):
    def __init__(self, input_size, hidden_size, compress_dim=4, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTMの定義、
        # batch_firstより、出力次元の最初がbatchとなる。
        # dropoutを用いて、内部状態のconnectionをdropすることより過学習を防ぐ。
        # Sequenceがかなり長いため、初期入力された情報の消失が激しいと想定される。それらの防止のため、bidirectionalのモデルを使う。
        self.cell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

        # より高次元の特徴量を抽出できるようにするため、classifierの手前で、compress_dim次元への線形圧縮を行う。
        self.compressor = nn.Linear(hidden_size * 2, compress_dim)

        # sentiment probabilityの出力層。
        self.classifier = nn.Linear(compress_dim, 1)

        # outputの範囲を[0, 1]とする。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 入力値xから出力までの流れを定義する。
        output, _ = self.cell(x)
        output = self.sigmoid(self.classifier(self.compressor(output[:, -1, :])))
        return output

    def extract_feature(self, x):
        # 入力値xから特徴量抽出までの流れを定義する。
        output, _ = self.cell(x)
        output = self.compressor(output[:, -1, :])
        return output


class FeatureCombinerHandler:
    def __init__(self, feature_combiner_params, store_dir):
        # モデル学習及び推論に用いるデバイスを定義する
        if torch.cuda.device_count() >= 1:
            self.device = "cuda"
            print("[+] Set Device: GPU")
        else:
            self.device = "cpu"
            print("[+] Set Device: CPU")

        # モデルのcheckpointや抽出した特徴量及びsentimentをstoreする場所を定義する。
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        # 上記で作成したfeaturecombinerを定義する。
        self.feature_combiner = FeatureCombiner(**feature_combiner_params).to(
            self.device
        )

        # 学習に用いるoptimizerを定義する。
        self.optimizer = torch.optim.Adam(
            params=self.feature_combiner.parameters(),
            lr=0.001,
        )

        # ロス関数の定義
        self.criterion = nn.BCELoss().to(self.device)

        # モデルのcheck pointが存在する場合、モデルをロードする
        self._load_model()

    # 学習に必要なデータ(並列のためbatch化されたもの)をサンプルする。
    def _sample_xy(self, data_type):
        assert data_type in ("train", "val")

        # data_typeより、data_typeに合致したデータを取得するようにしている。
        if data_type == "train":
            # dataloaderをiteratorとして定義し、next関数として毎時のデータをサンプルすることができる。
            # Iteratorは全てのデータがサンプルされると、StopIterationのエラーを発するが、そのようなエラーが出たとき、
            # Iteratorを再定義し、データをサンプルするようにしている。
            try:
                x, y = next(self.iterable_train_dataloader)
            except StopIteration:
                self.iterable_train_dataloader = iter(self.train_dataloader)
                x, y = next(self.iterable_train_dataloader)

        elif data_type == "val":
            try:
                x, y = next(self.iterable_val_dataloader)
            except StopIteration:
                self.iterable_val_dataloader = iter(self.val_dataloader)
                x, y = next(self.iterable_val_dataloader)

        return x.to(self.device), y.to(self.device)

    # モデルのパラメータをアップデートするロジック
    def _update_params(self, loss):
        # ロスから、gradientを逆伝播し、パラメータをアップデートする
        loss.backward()
        self.optimizer.step()

    # 学習されたfeature_combinerのパラメータをcheck_pointとしてstoreするロジック
    def _save_model(self, epoch):
        torch.save(
            self.feature_combiner.state_dict(),
            os.path.join(self.store_dir, f"{epoch}.ckpt"),
        )
        print(f"[+] Epoch: {epoch}, Model is saved.")

    # 学習されたcheckpointが存在す場合、feature_combinerにそのパラメータをロードするロジック
    def _load_model(self):
        # cudaで学習されたモデルなどを、cpu環境下でロードするときはこのパラメータが必要となる。
        params_to_load = {}
        if self.device == "cpu":
            params_to_load["map_location"] = torch.device("cpu")

        # .ckptファイルを探し、古い順から新しい順にソートする。
        check_points = glob(os.path.join(self.store_dir, "*.ckpt"))
        check_points = sorted(
            check_points,
            key=lambda x: int(x.split("/")[-1].replace(".ckpt", "")),
        )

        # check_pointが存在しない場合は、スキップする。
        if len(check_points) == 0:
            print("[!] No exists checkpoint")
            return

        # 複数個のchieck_pointが存在する場合、一番最新のものを使い、モデルのパラメータをロードする
        check_point = check_points[-1]
        self.feature_combiner.load_state_dict(torch.load(check_point, **params_to_load))
        print("[+] Model is loaded")

    # Datasetからdataloaderを定義するロジック
    def _build_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        # 上記3で作成したしたdatasetを定義する
        dataset = Dataset(
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # datasetのdataをiterableにロードできるよう、dataloaderを定義する、このとき、shuffle=Trueを渡すことで、データはランダムにサンプルされるようになる。
        return DataLoader(dataset=dataset, shuffle=True, **dataloader_params)

    # train用に、featuresとlabelsを渡し、datasetを定義し、dataloaderを定義するロジック
    def set_train_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        self.train_dataloader = self._build_dataloader(
            dataloader_params=dataloader_params,
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # dataloaderからiteratorを定義する
        # iteratorはnext関数よりデータをサンプルすることが可能となる。
        self.iterable_train_dataloader = iter(self.train_dataloader)

    # validation用に、featuresとlabelsを渡し、datasetを定義し、dataloaderを定義するロジック
    def set_val_dataloader(
        self, dataloader_params, weekly_features, weekly_labels, max_sequence_length
    ):
        self.val_dataloader = self._build_dataloader(
            dataloader_params=dataloader_params,
            weekly_features=weekly_features,
            weekly_labels=weekly_labels,
            max_sequence_length=max_sequence_length,
        )

        # dataloaderからiteratorを定義する
        # iteratorはnext関数よりデータをサンプルすることが可能となる。
        self.iterable_val_dataloader = iter(self.val_dataloader)

    # 学習ロジック
    def train(self, n_epoch):
        # n_epochの回数分、全学習データを複数回用いて学習する。
        for epoch in range(n_epoch):

            # 各々のepochごとのaverage lossを表示するため、lossをstoreするリストを定義する。
            train_losses = []
            test_losses = []

            # train_dataloaderの長さは、全ての学習データを一度用いるときの長さと同様である。
            # batchを組むと、その分train_dataloaderの長さは可変し、ちょうど一度全てのデータで学習できる長さを返す。
            for iter_ in tqdm(range(len(self.train_dataloader))):
                # パラメータをtrainableにするため、feature_combinerをtrainモードにする。
                self.feature_combiner.train()

                # trainデータをサンプルする。
                x, y = self._sample_xy(data_type="train")

                # feature_combinerに特徴量を入力し、sentiment scoreを取得する。
                preds = self.feature_combiner(x=x)

                # sentiment scoreとラベルとのロスを計算する。
                train_loss = self.criterion(preds, y.view(-1, 1))

                # 計算されたロスは、後ほどepochごとのdisplayに使用するため、storeしておく。
                train_losses.append(train_loss.detach())

                # lossから、gradientを逆伝播させ、パラメータをupdateする。
                self._update_params(loss=train_loss)

                # validation用のロースを計算する。
                # 毎回計算を行うとコストがかかってくるので、iter_毎5回ごとに計算を行う。
                if iter_ % 5 == 0:

                    # 学習を行わないため、feature_combinerをevalモードにしておく。
                    # evalモードでは、dropoutの影響を受けない。
                    self.feature_combiner.eval()

                    # 各パラメータごとのgradientを計算するとリソースが高まる。
                    # evaluationの時には、gradient情報を持たせないことで、メモリーの節約に繋がる。
                    with torch.no_grad():
                        # validationデータをサンプルする
                        x, y = self._sample_xy(data_type="val")

                        # feature_combinerに特徴量を入力し、sentiment scoreを取得する。
                        preds = self.feature_combiner(x=x)

                        # sentiment scoreとラベルとのロスを計算する。
                        test_loss = self.criterion(preds, y.view(-1, 1))

                        # 計算されたロスは、後ほどepochごとのdisplayに使用するため、storeしておく。
                        test_losses.append(test_loss.detach())

            # 毎epoch終了後、平均のロスをプリントする。
            print(
                f"epoch: {epoch}, train_loss: {np.mean(train_losses):.4f}, val_loss: {np.mean(test_losses):.4f}"
            )

            # 毎epoch終了後、モデルのパラメータをstoreする。
            self._save_model(epoch=epoch)

    # 特徴量から、合成特徴量を抽出するロジック
    def combine_features(self, features):
        # 学習を行わないため、feature_combinerをevalモードにしておく。
        self.feature_combiner.eval()

        # gradient情報を持たせないことで、メモリーの節約する。
        with torch.no_grad():

            # 特徴量をfeature_combinerのextract_feature関数に入力し、出力層手前の特徴量を抽出する。
            # 抽出するとき、tensorをcpu上に落とし、np.ndarray形式に変換する。
            return (
                self.feature_combiner.extract_feature(
                    x=torch.tensor(features, dtype=torch.float).to(self.device)
                )
                .cpu()
                .numpy()
            )

    # 特徴量から、翌週のsentimentを予測するロジック
    def predict_sentiment(self, features):
        # 学習を行わないため、feature_combinerをevalモードにしておく。
        self.feature_combiner.eval()

        # gradient情報を持たせないことで、メモリーの節約する。
        with torch.no_grad():

            # 特徴量をfeature_combinerに入力し、sentiment scoreを抽出する。
            # 抽出するとき、tensorをcpu上に落とし、np.ndarray形式に変換する。
            return (
                self.feature_combiner(
                    x=torch.tensor(features, dtype=torch.float).to(self.device)
                )
                .cpu()
                .numpy()
            )

    # weeklyグループされた特徴量を入力に、合成特徴量もしくは、sentiment scoreを抽出するロジック
    def generate_by_weekly_features(
        self, weekly_features, generate_target, max_sequence_length
    ):
        assert generate_target in ("features", "sentiment")
        generate_func = getattr(
            self,
            {"features": "combine_features", "sentiment": "predict_sentiment"}[
                generate_target
            ],
        )

        # グループごとに特徴量もしくは、sentiment scoreを抽出し、最終的に重ねて返すため、リストを作成する。
        outputs = []

        # ユニークな週indexを取得する。
        weeks = sorted(weekly_features.index.get_level_values(0).unique())

        for week in tqdm(weeks):
            # 各週ごとの特徴量を取得し、直近から、max_sequence_length分切る。
            features = weekly_features.xs(week, axis=0, level=0)[-max_sequence_length:]

            # 特徴量をモデルに入力し、合成特徴量もしくは、sentiment scoreを抽出し、outputsにappendする。
            # np.expand_dims(features, axis=0)を用いる理由は、特徴量合成機の入力期待値は、dimention0がbatchであるが、
            # featuresは、[1000, 768]の次元をもち、これらをunsqueezeし、[1, 1000, 768]に変換する必要がある。
            outputs.append(generate_func(features=np.expand_dims(features, axis=0)))

        # outputsを重ね、indexの情報とともにpd.DataFrame形式として返す。
        return pd.DataFrame(np.concatenate(outputs, axis=0), index=weeks)


class SentimentGenerator(object):
    article_columns = None
    device = None
    feature_extractor = None
    headline_feature_combiner_handler = None
    keywords_feature_combiner_handler = None
    punctuation_replace_dict = None
    punctuation_remove_list = None

    @classmethod
    def initialize(cls, base_dir="../model"):
        # 使用するcolumnをセットする。
        cls.article_columns = ["publish_datetime", "headline", "keywords"]

        # BERT特徴量抽出機をセットする。
        cls._set_device()
        cls._build_feature_extractor(base_dir)
        cls._build_tokenizer(base_dir)

        # LSTM特徴量合成機をセットする。
        cls.headline_feature_combiner_handler = FeatureCombinerHandler(
            feature_combiner_params={"input_size": 768, "hidden_size": 128},
            store_dir=f"{base_dir}/headline_features",
        )
        cls.keywords_feature_combiner_handler = FeatureCombinerHandler(
            feature_combiner_params={"input_size": 768, "hidden_size": 128},
            store_dir=f"{base_dir}/keywords_features",
        )

        # 置換すべき記号のdictionaryを作成する。
        JISx0208_replace_dict = {
            "髙": "高",
            "﨑": "崎",
            "濵": "浜",
            "賴": "頼",
            "瀨": "瀬",
            "德": "徳",
            "蓜": "配",
            "昻": "昂",
            "桒": "桑",
            "栁": "柳",
            "犾": "犹",
            "琪": "棋",
            "裵": "裴",
            "魲": "鱸",
            "羽": "羽",
            "焏": "丞",
            "祥": "祥",
            "曻": "昇",
            "敎": "教",
            "澈": "徹",
            "曺": "曹",
            "黑": "黒",
            "塚": "塚",
            "閒": "間",
            "彅": "薙",
            "匤": "匡",
            "冝": "宜",
            "埇": "甬",
            "鮏": "鮭",
            "伹": "但",
            "杦": "杉",
            "罇": "樽",
            "柀": "披",
            "﨤": "返",
            "寬": "寛",
            "神": "神",
            "福": "福",
            "礼": "礼",
            "贒": "賢",
            "逸": "逸",
            "隆": "隆",
            "靑": "青",
            "飯": "飯",
            "飼": "飼",
            "緖": "緒",
            "埈": "峻",
        }

        cls.punctuation_replace_dict = {
            **JISx0208_replace_dict,
            "《": "〈",
            "》": "〉",
            "『": "「",
            "』": "」",
            "“": '"',
            "!!": "!",
            "〔": "[",
            "〕": "]",
            "χ": "x",
        }

        # 取り除く記号リスト。
        cls.punctuation_remove_list = [
            "|",
            "■",
            "◆",
            "●",
            "★",
            "☆",
            "♪",
            "〃",
            "△",
            "○",
            "□",
        ]

    @classmethod
    def _set_device(cls):
        # 使用可能なgpuがある場合、そちらを利用し特徴量抽出を行う
        if torch.cuda.device_count() >= 1:
            cls.device = "cuda"
            print("[+] Set Device: GPU")
        else:
            cls.device = "cpu"
            print("[+] Set Device: CPU")

    @classmethod
    def load_feature_extractor(cls, model_dir, download=False, save_local=False):
        # 特徴量抽出のため事前学習済みBERTモデルを用いる。
        # ここでは、"cl-tohoku/bert-base-japanese-whole-word-masking"モデルを使用しているが、異なる日本語BERTモデルを用いても良い。
        target_model = "cl-tohoku/bert-base-japanese-whole-word-masking"
        save_dir = os.path.abspath(
            f"{model_dir}/transformers_pretrained/{target_model}"
        )
        if download:
            pretrained_model = target_model
        else:
            pretrained_model = save_dir

        feature_extractor = transformers.BertModel.from_pretrained(
            pretrained_model,
            return_dict=True,
            output_hidden_states=True,
        )

        if download and save_local:
            print(f"[+] save feature_extractor: {save_dir}")
            feature_extractor.save_pretrained(save_dir)

        return feature_extractor

    @classmethod
    def _build_feature_extractor(cls, model_dir, download=False):
        # 事前学習済みモデルを取得
        cls.feature_extractor = cls.load_feature_extractor(model_dir, download)

        # 使用するdeviceを指定
        cls.feature_extractor = cls.feature_extractor.to(cls.device)

        # 今回、学習は行わない。特徴量抽出のためなので、評価モードにセットする。
        cls.feature_extractor.eval()

        print("[+] Built feature extractor")

    @classmethod
    def load_bert_tokenizer(cls, model_dir, download=False, save_local=False):
        # BERTモデルの入力とするコーパスはそのBERTモデルが学習された時と同様の前処理を行う必要がある。
        # 今回使用する"cl-tohoku/bert-base-japanese-whole-word-masking"モデルは、
        # mecab-ipadic-NEologdによりトークナイズされ、その後Wordpiece subword encoderよりsubword化している。
        # Subwordとは形態素の類似な概念として、単語をより小さい意味のある単位に変換したものである。
        # transformersのBertJapaneseTokenizerは、その事前学習モデルの学習時と同様の前処理を簡単に使用することができる。
        # この章ではBertJapaneseTokenizerを利用し、トークナイズ及びsubword化を行う。
        target_model = "cl-tohoku/bert-base-japanese-whole-word-masking"
        save_dir = os.path.abspath(
            f"{model_dir}/transformers_pretrained/{target_model}"
        )
        if download:
            pretrained_model = target_model
        else:
            pretrained_model = save_dir
        bert_tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)
        if download and save_local:
            print(f"[+] save bert_tokenizer: {save_dir}")
            bert_tokenizer.save_pretrained(save_dir)

        return bert_tokenizer

    @classmethod
    def _build_tokenizer(cls, model_dir, download=False):
        # トークナイザーを取得
        cls.bert_tokenizer = cls.load_bert_tokenizer(model_dir, download)

        print("[+] Built bert tokenizer")

    @classmethod
    def load_articles(cls, path, start_dt=None, end_dt=None):
        # csvをロードする
        # headline、keywordsをcolumnとして使用。publish_datetimeをindexとして使用。
        articles = pd.read_csv(path)[cls.article_columns].set_index("publish_datetime")

        # str形式のdatetimeをpd.Timestamp形式に変換
        articles.index = pd.to_datetime(articles.index)

        # NaN値を取り除く
        articles = articles.dropna()

        # 必要な場合、使用するデータの範囲を指定する
        return articles[start_dt:end_dt]

    @classmethod
    def normalize_articles(cls, articles):
        articles = articles.copy()

        # 欠損値を取り除く
        articles = articles.dropna()

        for column in articles.columns:
            # スペース(全角スペースを含む)はneologdn正規化時に全て除去される。
            # ここでは、スペースの情報が失われないように、スペースを全て改行に書き換え、正規化後スペースに再変換する。
            articles[column] = articles[column].apply(lambda x: "\n".join(x.split()))

            # neologdnを使って正規化を行う。
            articles[column] = articles[column].apply(lambda x: neologdn.normalize(x))

            # 改行をスペースに置換する。
            articles[column] = articles[column].str.replace("\n", " ")

        return articles

    @classmethod
    def handle_punctuations_in_articles(cls, articles):
        articles = articles.copy()

        for column in articles.columns:
            # punctuation_remove_listに含まれる記号を除去する
            articles[column] = articles[column].str.replace(
                fr"[{''.join(cls.punctuation_remove_list)}]", ""
            )

            # punctuation_replace_dictに含まれる記号を置換する
            for replace_base, replace_target in cls.punctuation_replace_dict.items():
                articles[column] = articles[column].str.replace(
                    replace_base, replace_target
                )

            # unicode正規化を行う
            articles[column] = articles[column].apply(
                lambda x: unicodedata.normalize("NFKC", x)
            )

        return articles

    @classmethod
    def drop_remove_list_words(cls, articles, remove_list_words=["人事"]):
        articles = articles.copy()

        for remove_list_word in remove_list_words:
            # headlineもしくは、keywordsどちらかでremove_list_wordを含むニュース記事のindexマスクを作成。
            drop_mask = articles["headline"].str.contains(remove_list_word) | articles[
                "keywords"
            ].str.contains(remove_list_word)

            # remove_list_wordを含まないニュースだけに精製する。
            articles = articles[~drop_mask]

        return articles

    @classmethod
    def build_inputs(cls, texts, max_length=512):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for text in texts:
            encoded = cls.bert_tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )

            input_ids.append(encoded["input_ids"])
            token_type_ids.append(encoded["token_type_ids"])
            attention_mask.append(encoded["attention_mask"])

        # torchモデルに入力するためにはtensor形式に変え、deviceを指定する必要がある。
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(cls.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(cls.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(cls.device)

        return input_ids, token_type_ids, attention_mask

    @classmethod
    def generate_features(cls, input_ids, token_type_ids, attention_mask):
        output = cls.feature_extractor(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        features = output["hidden_states"][-2].mean(dim=1).cpu().detach().numpy()

        return features

    @classmethod
    def generate_features_by_texts(cls, texts, batch_size=2, max_length=512):
        n_batch = math.ceil(len(texts) / batch_size)

        features = []
        for idx in tqdm(range(n_batch)):
            input_ids, token_type_ids, attention_mask = cls.build_inputs(
                texts=texts[batch_size * idx : batch_size * (idx + 1)],
                max_length=max_length,
            )

            features.append(
                cls.generate_features(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
            )

        features = np.concatenate(features, axis=0)

        # 抽出した特徴量はnp.ndarray形式となっており、これらは、日付の情報を失っているため、pd.DataFrame形式に変換する。
        return pd.DataFrame(features, index=texts.index)

    @classmethod
    def _build_weekly_group(cls, df):
        # index情報から、(year, week)の情報を得る。
        return pd.Series(list(zip(df.index.year, df.index.week)), index=df.index)

    @classmethod
    def build_weekly_features(cls, features, boundary_week):
        assert isinstance(boundary_week, tuple)

        weekly_group = cls._build_weekly_group(df=features)
        features = features.groupby(weekly_group).apply(lambda x: x[:])

        train_features = features[features.index.get_level_values(0) <= boundary_week]
        test_features = features[features.index.get_level_values(0) > boundary_week]

        return {"train": train_features, "test": test_features}

    @classmethod
    def generate_lstm_features(
        cls,
        article_path,
        start_dt=None,
        boundary_week=(2020, 26),
        target_feature_types=None,
    ):
        # target_feature_typesが指定されなかったらデフォルト値設定
        dfault_target_feature_types = [
            "headline",
            "keywords",
        ]
        if target_feature_types is None:
            target_feature_types = dfault_target_feature_types
        # feature typeが想定通りであることを確認
        assert set(target_feature_types).issubset(dfault_target_feature_types)

        # ニュースデータをロードする。
        articles = cls.load_articles(start_dt=start_dt, path=article_path)

        # 前処理を行う。
        articles = cls.normalize_articles(articles)
        articles = cls.handle_punctuations_in_articles(articles)
        articles = cls.drop_remove_list_words(articles)

        # headlineとkeywordsの特徴量をdict型で返す。
        lstm_features = {}

        for feature_type in target_feature_types:
            # コーパス全体のBERT特徴量を抽出する。
            features = cls.generate_features_by_texts(texts=articles[feature_type])

            # feature_typeに合致するfeature_combiner_handlerをclsから取得する。
            feature_combiner_handler = {
                "headline": cls.headline_feature_combiner_handler,
                "keywords": cls.keywords_feature_combiner_handler,
            }[feature_type]

            # 特徴量を週毎のグループ化する。
            weekly_features = cls.build_weekly_features(features, boundary_week)["test"]

            # Sentiment scoreを抽出する。
            lstm_features[
                f"{feature_type}_features"
            ] = feature_combiner_handler.generate_by_weekly_features(
                weekly_features=weekly_features,
                generate_target="sentiment",
                max_sequence_length=10000,
            )

        return lstm_features
