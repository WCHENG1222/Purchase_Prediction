# Online Purchase Intention Prediction

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from functools import reduce
import pathlib


def main():
    # X:
    # | Administrative | Administrative_Duration | Informational | Informational_Duration |
    # | ProductRelated | ProductRelated_Duration | BounceRates | ExitRates | PageValues |
    # | SpecialDay | Month | OperatingSystems | Browser | Region | TrafficType | VisitorType | Weekend |
    # Y:
    # | Revenue |

    # dataset source: https://www.kaggle.com/henrysue/online-shoppers-intention
    behavior_data = pd.read_csv(r"online_shoppers_intention.csv", sep=',', encoding='ISO-8859-1')

    """data wrangling"""
    # behavior_data.info(memory_usage='deep')
    # bool type to 0 or 1
    behavior_data[['Weekend', 'Revenue']] = behavior_data[['Weekend', 'Revenue']].astype(int)

    # transform string columns to dummy variable
    str_col = behavior_data.columns[behavior_data.applymap(type).eq(str).any()]
    dummies = pd.get_dummies(behavior_data[str_col])
    behavior_data = pd.concat([behavior_data, dummies], axis=1).drop(str_col, axis=1)

    # split dependent(y) and independent(x) variable
    y = behavior_data['Revenue']  # 1: 15.5% , 0: 84.5%
    x = behavior_data.drop(['Revenue'], axis=1)

    # split train(70%) and test(30%) set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=99)

    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # feature selection by Mutual Information
    selector = SelectKBest(mutual_info_classif, k=10)
    x_train = selector.fit_transform(x_train, y_train)
    x_test = selector.transform(x_test)
    select_feat = x.columns[selector.get_support()]
    # select_feat_score = selector.scores_

    # over-sampling by SMOTE(Synthetic Minority Oversampling Technique)
    x_train, y_train = SMOTE(random_state=99).fit_resample(x_train, y_train)

    """model"""
    lgr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=99, verbose=1)
    xgb = XGBClassifier(n_estimators=100, n_jobs=8, learning_rate=0.5, random_state=99, eval_metric='auc',
                        use_label_encoder=False)
    lgr_profit = model_profit(lgr, x_train, x_test, y_train, y_test,
                              customer_consumption=800, product_cost_pct=0.3, marketing_cost_pct=0.1)
    rf_profit = model_profit(rf, x_train, x_test, y_train, y_test,
                             customer_consumption=800, product_cost_pct=0.3, marketing_cost_pct=0.1)
    xgb_profit = model_profit(xgb, x_train, x_test, y_train, y_test,
                              customer_consumption=800, product_cost_pct=0.3, marketing_cost_pct=0.1)

    # profit evaluation
    profit_trade_off = [lgr_profit, rf_profit, xgb_profit]
    profit_trade_off = reduce(lambda left, right:
                              pd.merge(left, right, on=['proba_threshold', 'full_marketing_profit']), profit_trade_off)
    print(profit_trade_off)

    #  feature analysis
    feature_analysis = pd.merge(feature_score(lgr, select_feat), feature_score(rf, select_feat), on=['feature'])

    """output to csv"""
    profit_trade_off.to_csv(pathlib.Path.cwd() / 'profit_trade_off.csv', header=True, mode='a', sep=',', index=False,
                            encoding='utf8')
    feature_analysis.to_csv(pathlib.Path.cwd() / 'feature_analysis.csv', header=True, mode='a', sep=',', index=False,
                            encoding='utf8')


def model_training(model, x_train, x_test, y_train, y_test):
    # training
    model.fit(x_train, y_train)
    model_name = model.__class__.__name__

    # prediction
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    # evaluation
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # cls_report = classification_report(y_test, y_pred)
    return model_name, y_pred_proba, y_test, y_pred


def model_profit(model, x_train, x_test, y_train, y_test,
                 customer_consumption, product_cost_pct, marketing_cost_pct):
    """
    Model Profit Trade Off -> | proba_threshold | full_marketing_profit | ml_marketing_profit |
                              | ml_precision | ml_recall | ml_f1 | ml_auc |
    """
    model_name, y_pred_proba = model_training(model, x_train, x_test, y_train, y_test)[:2]

    # calc profit
    product_cost = customer_consumption * product_cost_pct
    marketing_cost = customer_consumption * marketing_cost_pct

    # iterate threshold
    profit_trade_off_list = []
    pred_proba = pd.DataFrame(y_pred_proba[:, 1], columns=['pred_proba_1'], index=y_test.index)
    for thres in np.linspace(start=0, stop=1, num=101, endpoint=True):
        pred_proba['pred_thres'] = (
            np.select(
                condlist=[pred_proba['pred_proba_1'] < thres, pred_proba['pred_proba_1'] >= thres],
                choicelist=[0, 1])
        )
        conf_thres = confusion_matrix(y_test, pred_proba['pred_thres'])

        # profit evaluation
        full_marketing_profit = (
                (customer_consumption * conf_thres[1, :].sum()) - (product_cost * conf_thres[1, :].sum())
                - (marketing_cost * conf_thres.sum())
        )
        ml_marketing_profit = (
                (customer_consumption * conf_thres[1, 1]) - (product_cost * conf_thres[1, 1])
                - (marketing_cost * conf_thres[:, 1].sum())
        )

        # metrics evaluation
        precision = precision_score(y_test, pred_proba['pred_thres'], zero_division=0)
        recall = recall_score(y_test, pred_proba['pred_thres'], zero_division=0)
        f1 = f1_score(y_test, pred_proba['pred_thres'])
        auc = roc_auc_score(y_test, pred_proba['pred_thres'])

        profit_trade_off_list.append([thres, full_marketing_profit, ml_marketing_profit, precision, recall, f1, auc])

    profit_trade_off = pd.DataFrame(
        profit_trade_off_list,
        columns=['proba_threshold', 'full_marketing_profit', f'{model_name}_marketing_profit',
                 f'{model_name}_precision', f'{model_name}_recall', f'{model_name}_f1', f'{model_name}_auc'])
    return profit_trade_off


def feature_score(model, feat_col):
    """
    Feature Score -> | feature_name | ml_feature_importance |
    """
    model_name = model.__class__.__name__
    if model_name == 'LogisticRegression':
        # logistic regression odds ratio: while x increase, y purchase event / y not purchase event
        # variation of x -> y
        odds_ratio = np.exp(model.coef_[0])-1
        feat_imp = odds_ratio
    else:
        feat_imp = model.feature_importances_
    feat_imp_df = pd.DataFrame({'feature': feat_col, f'{model_name}_importance': feat_imp})
    return feat_imp_df


if __name__ == '__main__':
    main()
