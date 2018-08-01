#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class GradientBoostingEncoder(OneHotEncoder):
    '''
    GradientBoostingを用いて特徴量を離散化・拡張する

    Parameters:
    -----
    gbr: sklearn.ensemble.GradientBoostingRegressor/Classifier
        fit() 済のGradientBoostingRegressor/Classifier のインスタンス
    feature_names: array of str or None
        GradientBoostingRegressorに与えた特徴量の名前の配列
        None の場合は 'feature_[feature_id]' という名前になる
    prefix: str (default: 'gbr')
        特徴量の名前の先頭に付けるprefix文字列
        classes_ が [prefix]_[tree_id]_[leaf_id]になる

    Attributes:
    -----
    classes_: array of str
        変換された特徴量の名前（[prefix]_[tree_id]_[leaf_id]のリスト）
    class_maps_: dictionary of (str, str)
        classes_ と実際の特徴量から生成された名前の変換辞書
    '''
    def __init__(self, gbr, feature_names=None, prefix='gbr'):
        super(GradientBoostingEncoder, self).__init__(
            n_values='auto',
            categorical_features='all',
            sparse=False,
            handle_unknown='ignore'
        )
        self.gbr = gbr
        if feature_names is None:
            self.feature_names = [
                'feature_%d' for x in range(len(gbr.feature_importances_))
            ]
        else:
            self.feature_names = feature_names
        self.prefix = prefix

    def _trace_tree_recursively(
        self,
        tree_id,
        tree,
        features_list,
        features_dict,
        node_id=0,
        conds=[]
    ):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child == -1:
            # leaf node の場合
            leaf_name = '_'.join([self.prefix, str(tree_id), str(node_id)])
            features_list.append(leaf_name)
            features_dict[leaf_name] = ' & '.join(conds)
        else:
            # 配下に node がある場合
            # その node で判定に用いた特徴量
            feature = self.feature_names[tree.feature[node_id]]
            # その特徴量で左右に分岐する閾値
            threshold = round(tree.threshold[node_id], 2)
            # 左と右に行く条件
            left_cond = '%s<=%s' % (feature, threshold)
            right_cond = '%s>%s' % (feature, threshold)
            # 再帰的に配下の node を呼び出す
            self._trace_tree_recursively(
                tree_id,
                tree,
                features_list,
                features_dict,
                node_id=left_child,
                conds=conds + [left_cond]
            )
            self._trace_tree_recursively(
                tree_id,
                tree,
                features_list,
                features_dict,
                node_id=right_child,
                conds=conds + [right_cond]
            )

    def fit_transform(self, X, y=None):
        # [prefix]_[node_id]_[leaf_id]の名前が入るリスト
        gbfeatures_list = list()
        # leaf の名前とleafに至る条件が格納される辞書
        gbfeatures_dict = dict()
        # データが到達したleaf_idが入るリスト
        gbfeatures_data = list()
        # tree_idの初期値
        tree_id = 0
        # GradientBoostingRegressorのestimators_ は
        # DecisionTreeRegressorの配列の配列（[n_estimators, 1]）
        for decision_tree_regressor in self.gbr.estimators_.T[0]:
            gbfeatures_data.append(
                decision_tree_regressor.apply(X)
            )
            # 各木について、leafの名前とleafに至る条件を得る
            # （root から再帰的に辿るため、node_id と conds は指定しない（＝デフォルトの値を使う））
            self._trace_tree_recursively(
                tree_id,
                decision_tree_regressor.tree_,
                gbfeatures_list,
                gbfeatures_dict
            )
            tree_id += 1
        self.classes_ = gbfeatures_list
        self.class_maps_ = gbfeatures_dict
        gbfeatures = np.array(gbfeatures_data).T
        return super(GradientBoostingEncoder, self).fit_transform(gbfeatures)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        gbfeatures_data = list()
        for decision_tree_regressor in self.gbr.estimators_.T[0]:
            gbfeatures_data.append(
                decision_tree_regressor.apply(X)
            )
        gbfeatures = np.array(gbfeatures_data).T
        return super(GradientBoostingEncoder, self).transform(gbfeatures)
