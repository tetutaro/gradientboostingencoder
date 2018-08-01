# Gradient Boosting Encoder

Facebookがオンライン広告のCTRを予測するために作ったモデルをヒントに、scikit-learnのGradientBoostingから特徴量を離散化＆かけ合わせするエンコーダ

He, Xinran, et al. "Practical lessons from predicting clicks on ads at facebook." Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 2014.

GradientBoostingRegressorにもGradientBoostingClassifierにも対応しているので、RegressionにもClassificationにも使える

Facebookの論文のうち、Encoder部分だけ実装したので、逆にRegressor/Classifierには好きなものを使える

## インストール方法

`> pip install git+https://github.com/tetutaro/gradientboostingencoder`

## 使い方の例

あくまで例なので、モデルの作り方は適当

```
from sklearn.datasets import load_wine
wine = load_wine()

from sklearn.ensemble import GradientBoostingClassifier
from gradientboostingencoder import GradientBoostingEncoder

# 例として使うデータがClassificationのデータなのでClassifierを使うが
# RegressionのデータならばRegressorを使うことも可能

cls = GradientBoostingClassifier(
    n_estimators=3, max_depth=2, random_state=14
).fit(wine.data, wine.target)
enc = GradientBoostingEncoder(
    cls, feature_names=wine.feature_names, prefix='gbc'
).fit(wine.data)

# depth 2 の木なので、ひとつあたりMAX 4 個の特徴量、それが 3 本出来るはず
# よって、最大12個の特徴量に変換される
# 便宜的に特徴量の名前は (prefix)_(nodeのID)_..._(leafのID) というものにする
# prefix はデフォルトで "gbr"
# classes_ にその配列が格納される

enc.classes_

# feature_namesを与えれば、その名前と「どの値以上/未満で1に変換したか」を
# 上記特徴量名との辞書の形で返す

enc.class_maps_

enced_data = enc.transform(wine.data)

# 後はこのデータを用いて好きにregression/classificationして良い
# 例としてRandomForestのimportanceを出すことにする

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    random_state=14
).fit(enced_data, wine.target)

# 予測精度の表示

from sklearn.metrics import accuracy_score

accuracy_score(wine.target, rf.predict(enced_data))

# importanceが高い特徴量Top3の表示

import pandas as pd

pd.DataFrame(
    [(x, y) for x, y in zip(
        [enc.class_maps_[z] for z in enc.classes_],
        rf.feature_importances_
    )],
    columns=['feature', 'importance']
).sort_values(
    by='importance', ascending=False
).reset_index(
    drop=True
).iloc[:3,]
```
