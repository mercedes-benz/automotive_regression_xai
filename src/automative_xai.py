from xai_classes.xai_pfi import PFI
from xai_classes.xai_ifi import IFI
from xai_classes.xai_lime import LIME
from xai_classes.xai_ale import ALE
from xai_classes.xai_shap import SHAP
import matplotlib.pyplot as plt


class Automotive_XAI:
    def __init__(self, model, features, feature_name, X, y):
        self.model = model
        self.X = X
        self.features = features
        self.y = y
        self.feature_name = feature_name

    def do_PFI(self):
        """Perform and plot Permutation Feature Importance (PFI)"""
        pfi_explainer = PFI(self.model, self.features)
        pfi_explainer.print_importances(self.X[self.features], self.y, "Test")
        pfi_explainer.plot_importances("Test")

        sorted_importances = pfi_explainer.get_feature_importances()
        print("\nSorted Feature Importances (PFI):")
        for importance, feature in sorted_importances:
            print(f"{feature}: {importance:.3f}")

    def do_IFI(self):
        """Perform and plot Individual Feature Importance (IFI)"""
        ifi_explainer = IFI(self.model, self.features)
        ifi_explainer.print_importances()
        ifi_explainer.plot_importances()

    def do_LIME(self):
        """Perform LIME explanation"""
        lime_exp = LIME(self.model, self.X[self.features], self.y, self.features)
        lime_exp.set_local_index(self.y)
        lime_exp.explain(self.X[self.features], self.y)

    def do_SHAP(self):
        """Perform and plot SHAP values"""
        shap_analysis = SHAP(model=self.model, features=self.features)
        shap_values, X_sample, X_background = shap_analysis.compute_shap_values(
            self.X[self.features]
        )
        shap_analysis.generate_plots(
            shap_values, X_sample, X_background, feature_name=self.feature_name
        )

    def do_ALE(self):
        """Perform and plot ALE (Accumulated Local Effects)"""
        ale_plotter = ALE(model=self.model, features=self.features)
        ale_plotter.plot_ale(X=self.X[self.features], feature_name=self.feature_name)
