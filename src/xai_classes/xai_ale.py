from PyALE import ale
import matplotlib.pyplot as plt
import os


class ALE:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.output_dir = os.path.join("results", "ALE")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_ale(self, X, feature_name, grid_size=50, include_CI=False, figsize=(8, 6)):
        # Ensure the feature is valid
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' is not in the feature list.")

        # Compute ALE
        ale_eff = ale(
            X=X,
            model=self.model,
            feature=[feature_name],
            grid_size=grid_size,
            include_CI=include_CI,
        )

        # Plot ALE
        plt.figure(figsize=figsize)
        ale_eff.plot()
        plt.title(f"1D ALE Plot - {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel("ALE Effect on Predictions")

        # Save the plot
        plot_path = os.path.join(self.output_dir, f"{feature_name}_ale_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()  # Release memory
        print(f"ALE plot saved to {plot_path}")
