import os
import random
import shap
import matplotlib.pyplot as plt


class SHAP:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.output_dir = os.path.join("results", "SHAP")
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_shap_values(self, X, background_sample_size=100, shap_sample_size=500):
        # Sample a smaller subset of data
        X_sample = X.sample(n=shap_sample_size, random_state=42)

        # Select a subset for SHAP background distribution
        X_background = shap.utils.sample(X_sample, background_sample_size)

        # Create a SHAP explainer
        explainer = shap.Explainer(self.model.predict, X_background)
        shap_values = explainer(X_sample)

        return shap_values, X_sample, X_background

    def generate_plots(
        self, shap_values, X_sample, background_data, feature_name="carat"
    ):
        # Summary Plot
        shap.summary_plot(shap_values, X_sample)
        plt.savefig(os.path.join(self.output_dir, "shap_summary_plot.png"))
        plt.close()

        # Partial Dependence Plot
        sample_ind = random.randint(0, len(X_sample) - 1)
        plt.figure()
        shap.partial_dependence_plot(
            feature_name,
            self.model.predict,
            background_data,
            model_expected_value=True,
            feature_expected_value=True,
            ice=False,
            shap_values=shap_values[sample_ind : sample_ind + 1, :],
        )
        plt.savefig(os.path.join(self.output_dir, f"{feature_name}_par_dep_plot.png"))
        plt.close()

        # SHAP Bar Plot
        print("Generating SHAP Bar Plot...")
        plt.figure()
        shap.plots.bar(shap_values)
        plt.savefig(os.path.join(self.output_dir, "shap_bar_plot.png"))
        plt.close()

        # SHAP Waterfall Plot
        print(f"Generating SHAP Waterfall Plot for sample index {sample_ind}...")
        plt.figure()
        shap.plots.waterfall(shap_values[sample_ind])
        plt.savefig(os.path.join(self.output_dir, "shap_waterfall_plot.png"))
        plt.close()
