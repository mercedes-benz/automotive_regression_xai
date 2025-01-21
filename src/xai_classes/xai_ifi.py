import os
import numpy as np
import pandas as pd
import matplotlib.pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sklearn
import time


class IFI:
    def __init__(self, model, features, model_id="linear_regressor"):
        self.ifi_path = os.path.join("results", "IFI")
        os.makedirs(self.ifi_path, exist_ok=True)

        self.regr = model
        self.features = features
        self.model_id = model_id
        self.feature_importances = None

    def calculate_importances(self):
        if (
            type(self.regr) == sklearn.ensemble._forest.ExtraTreesRegressor
            or sklearn.ensemble.RandomForestRegressor
            or sklearn.ensemble.GradientBoostingRegressor
        ):
            start_time = time.time()
            self.feature_importances = self.regr.feature_importances_
            print(
                f"--- {(time.time() - start_time)}seconds to calculate impurity feature importances ---"
            )

    def print_importances(self):
        if type(self.feature_importances) != np.ndarray:
            self.calculate_importances()

        with open(
            os.path.join(
                self.ifi_path, f"{self.model_id}-ImpurityFeatureImportance.txt"
            ),
            "w",
        ) as f:
            for k, v in sorted(
                zip(self.feature_importances, self.features),
                reverse=True,
            ):
                f.write(f"{k:.3f} {v:20}\n")

    def plot_importances(self, max_display=6):
        if type(self.feature_importances) != np.ndarray:
            self.calculate_importances()

        ifi_data = {
            "features": self.features,
            "feature importances": self.feature_importances,
        }
        ifi_df = pd.DataFrame(ifi_data)
        ifi_df.sort_values(by=["feature importances"], ascending=False, inplace=True)
        ifi_df = ifi_df[0:max_display][:]
        plt.figure(figsize=(12, 8))
        plt.title(f"{self.model_id} - Impurity Feature Importance", fontsize=12)
        img = sns.barplot(
            x=ifi_df["feature importances"],
            y=ifi_df["features"],
            palette="ch:s=-.25,rot=.25",
        )
        img.set_yticklabels(img.get_yticklabels(), fontsize=12)
        img.set_xticklabels(img.get_xticklabels(), fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.ifi_path,
                f"{self.model_id}-ImpurityFeatureImportances.png",
            ),
            dpi=300,
        )
        plt.clf()

    def get_feature_importances(self):
        return sorted(
            zip(
                [float(x) for x in self.feature_importances.importances_mean],
                self.features,
            ),
            reverse=True,
        )
