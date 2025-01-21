import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import time


class PFI:
    def __init__(self, model, features, model_id="linear_regressor"):
        self.pfi_path = os.path.join("results", "PFI")
        os.makedirs(self.pfi_path, exist_ok=True)

        self.regr = model
        self.features = features
        self.model_id = model_id
        self.feature_importances = None
        self.pfi_features = None

    def calculate_importances(self, X, y, dataset):

        print("Calculating permutation feature importance...")
        start_time = time.time()
        self.feature_importances = permutation_importance(
            self.regr, X, y, random_state=0
        )
        print(
            f"--- {time.time() - start_time:.2f} seconds to calculate permutation feature importances on {dataset} dataset ---"
        )

    def print_importances(self, X, y, dataset):
        if self.feature_importances is None:
            self.calculate_importances(X, y, dataset)

        with open(
            os.path.join(
                self.pfi_path, f"{self.model_id}-PermutationFeatureImportance.txt"
            ),
            "w",
        ) as f:
            for i in self.feature_importances.importances_mean.argsort()[::-1]:
                f.write(
                    f"{self.feature_importances.importances_mean[i]:.3f}"
                    f" +/- {self.feature_importances.importances_std[i]:.3f}"
                    f" {self.features[i]:20}\n"
                )
        print("Feature importances saved to file.")

    def plot_importances(self, dataset, max_display=6):
        if self.feature_importances is None:
            raise ValueError("Feature importances have not been calculated yet.")

        pfi_data = {
            "features": self.features,
            "feature importances": self.feature_importances.importances_mean,
            "feature importances std": self.feature_importances.importances_std,
        }

        pfi_df = pd.DataFrame(pfi_data)
        pfi_df.sort_values(by=["feature importances"], ascending=False, inplace=True)
        pfi_df = pfi_df[:max_display]
        self.pfi_features = pfi_df["features"].tolist()

        plt.figure(figsize=(12, 8))
        plt.title(
            f"{self.model_id} - Permutation Feature Importance - {dataset}", fontsize=12
        )
        sns.barplot(
            x=pfi_df["feature importances"],
            y=pfi_df["features"],
            palette="ch:s=-.25,rot=.25",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.pfi_path,
                f"{self.model_id}-PermutationFeatureImportances-{dataset}.png",
            ),
            dpi=300,
        )
        plt.clf()
        print(f"Feature importance plot saved to {self.pfi_path}.")

    def get_feature_importances(self):
        if self.feature_importances is None:
            raise ValueError("Feature importances have not been calculated yet.")

        return sorted(
            zip(
                [float(x) for x in self.feature_importances.importances_mean],
                self.features,
            ),
            reverse=True,
        )
