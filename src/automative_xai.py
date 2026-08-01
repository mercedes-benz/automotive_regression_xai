from xai_classes.xai_pfi import PFI
from xai_classes.xai_ifi import IFI
from xai_classes.xai_lime import LIME
from xai_classes.xai_ale import ALE
from xai_classes.xai_shap import SHAP
import matplotlib.pyplot as plt


class Automotive_XAI:
    """
    A class to perform Explainable AI (XAI) analysis on an automotive predictive model.
    Includes methods for PFI, IFI, LIME, SHAP, and ALE explanations.
    """

    def __init__(self, model, features, feature_name, X, y):
        """
        Initialize the Automotive_XAI class.

        Parameters:
        - model: Trained ML model to be explained.
        - features: List of feature names used in the model.
        - feature_name: Specific feature for ALE and SHAP analysis.
        - X: Input dataset (pandas DataFrame or NumPy array).
        - y: Target variable (pandas Series or NumPy array).
        """
        self.model = model
        self.X = X
        self.features = features
        self.y = y
        self.feature_name = feature_name

    def do_PFI(self):
        """
        Compute and visualize Permutation Feature Importance (PFI).
        PFI shuffles feature values to measure their impact on model predictions.
        
        Returns:
        - sorted_importances: List of tuples (importance, feature).
        """
        pfi_explainer = PFI(self.model, self.features)
        pfi_explainer.print_importances(self.X[self.features], self.y, "Test")
        pfi_explainer.plot_importances("Test")

        sorted_importances = pfi_explainer.get_feature_importances()
        print("\nSorted Feature Importances (PFI):")
        for importance, feature in sorted_importances:
            print(f"{feature}: {importance:.3f}")

        return sorted_importances

    def do_IFI(self):
        """
        Compute and visualize Individual Feature Importance (IFI).
        IFI assesses the impact of individual feature values on model predictions.
        
        Returns:
        - ifi_values: Feature importance values.
        """
        ifi_explainer = IFI(self.model, self.features)
        ifi_explainer.print_importances()
        ifi_explainer.plot_importances()
        return ifi_explainer.get_importances()

    def do_LIME(self):
        """
        Perform LIME (Local Interpretable Model-agnostic Explanations) analysis.
        LIME explains individual predictions by approximating the model locally.
        
        Returns:
        - lime_exp: LIME explanation object.
        """
        lime_exp = LIME(self.model, self.X[self.features], self.y, self.features)
        
        # Use an index from the dataset instead of directly setting it to self.y
        sample_index = 0  # Modify as needed
        lime_exp.set_local_index(sample_index)

        lime_exp.explain(self.X[self.features], self.y)
        return lime_exp

    def do_SHAP(self):
        """
        Compute and visualize SHAP (SHapley Additive exPlanations) values.
        SHAP explains model predictions by assigning contributions to each feature.

        Returns:
        - shap_values: Computed SHAP values.
        - X_sample: Sample input data.
        - X_background: Background dataset for SHAP calculations.
        """
        shap_analysis = SHAP(model=self.model, features=self.features)
        shap_values, X_sample, X_background = shap_analysis.compute_shap_values(
            self.X[self.features]
        )
        shap_analysis.generate_plots(
            shap_values, X_sample, X_background, feature_name=self.feature_name
        )

        return shap_values, X_sample, X_background

    def do_ALE(self):
        """
        Compute and visualize Accumulated Local Effects (ALE).
        ALE captures the effect of a feature while considering dependencies with other features.

        Returns:
        - ale_plot: Generated ALE plot.
        """
        ale_plotter = ALE(model=self.model, features=self.features)
        ale_plot = ale_plotter.plot_ale(X=self.X[self.features], feature_name=self.feature_name)
        return ale_plot
