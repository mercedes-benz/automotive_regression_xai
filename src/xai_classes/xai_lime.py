import os
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sklearn
from lime.lime_tabular import LimeTabularExplainer


class LIME:
    def __init__(self, model, X, y, features, model_id="linear_regressor"):
        self.regr = model
        self.X_trn = X
        self.y_trn = y
        self.features = features
        self.model_id = model_id
        self.index_list = []
        self.lime_explainer = self._create_lime_explainer()

        self.lime_path = os.path.join("results", "LIME")
        os.makedirs(self.lime_path, exist_ok=True)

    def set_local_index(self, y):
        y_np = y.to_numpy()
        max_index = np.argmax(y_np)
        percentile_10 = np.percentile(y_np, 10)
        closest_val = y_np[np.abs(y_np - percentile_10).argmin()]
        min_index = np.where(y_np == closest_val)[0][0]
        self.index_list = [min_index, max_index]
        return None

    def _create_lime_explainer(self):
        return LimeTabularExplainer(
            self.X_trn.values,
            mode="regression",
            training_labels=self.y_trn.values,
            feature_names=self.features,
            feature_selection="auto",
        )

    def explain(self, X_exp, y_exp) -> object:
        X_exp = X_exp.values
        y_exp = y_exp.values

        if not self.index_list:
            raise ValueError(
                "Index list is empty. Please set `self.index_list` using `set_local_index` before calling `explain`."
            )

        explanations = []
        for index in self.index_list:
            print(f"---Explaining LIME local prediction at {index}.---")

            exp = self.lime_explainer.explain_instance(
                X_exp[index], self.regr.predict, num_features=6
            )

            explanations.append(exp)

            print(
                f"Prediction at X[{index}]: {self.regr.predict(X_exp[index].reshape(1, -1))}"
            )
            print(f"Actual at X[{index}]: {y_exp[index]}")

            html_obj = exp.as_html()
            with open(
                os.path.join(
                    self.lime_path,
                    f"{self.model_id}-{index}-LIME.html",
                ),
                "w",
            ) as f:
                for k in html_obj:
                    f.write(k)

        return explanations
