import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xai_classes.xai_pfi import PFI
from xai_classes.xai_ifi import IFI
from xai_classes.xai_lime import LIME
from xai_classes.xai_ale import ALE
from xai_classes.xai_shap import SHAP


class SampleDataset:
    def __init__(self, url):
        self.url = url
        self.data = None
        self.X_raw = None
        self.X = None
        self.y = None
        self.features = None
        self.processed = False

    def load_and_preprocess(self):
        """Load and preprocess the dataset."""
        # Load dataset
        self.data = pd.read_csv(self.url)

        # Separate features and target variable
        self.X_raw = self.data.loc[:, ~self.data.columns.str.contains("price")].copy()
        self.y = self.data["price"].copy()

        self.X = self.X_raw.copy()

        # Encode categorical features
        self.X["cut"] = self.X["cut"].astype(
            pd.api.types.CategoricalDtype(
                categories=["Fair", "Good", "Very Good", "Premium", "Ideal"],
                ordered=True,
            )
        )
        self.X["clarity"] = self.X["clarity"].astype(
            pd.api.types.CategoricalDtype(
                categories=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
                ordered=True,
            )
        )

        # Generate numeric codes for categorical features
        self.X["cut_code"] = self.X["cut"].cat.codes
        self.X["clarity_code"] = self.X["clarity"].cat.codes

        # One-hot encode the 'color' feature
        one_hot_encoder = OneHotEncoder().fit(self.X[["color"]])
        coded_feature = self.onehot_encode(self.X[["color"]], ohe=one_hot_encoder)
        self.X = pd.concat([self.X, coded_feature], axis=1)

        # Define the list of features
        self.features = [
            "carat",
            "cut_code",
            "clarity_code",
            "depth",
            "table",
            "x",
            "y",
            "z",
        ]
        self.features += coded_feature.columns.to_list()

        self.processed = True

    def onehot_encode(self, feat, ohe):
        """Perform one-hot encoding for categorical features."""
        col_names = ohe.categories_[0]
        feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
        feat_coded.columns = col_names
        return feat_coded

    def get_data(self):
        """Return the processed data."""
        if not self.processed:
            raise ValueError(
                "Data has not been preprocessed yet. Call load_and_preprocess() first."
            )
        feature_name = "carat"
        return self.X, self.y, self.features, feature_name
