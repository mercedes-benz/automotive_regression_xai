from sample_dataset import SampleDataset
from sklearn.ensemble import RandomForestRegressor
from automative_xai import Automotive_XAI


def test_dataset():
    # Create the SampleDataset instance
    dataset = SampleDataset(
        "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv"
    )

    # Load and preprocess the dataset
    dataset.load_and_preprocess()

    # Get the processed data
    X, y, features, feature_name = dataset.get_data()

    # Train RandomForest model
    model = RandomForestRegressor(random_state=1345)
    model.fit(X[features], y)

    # Instantiate the Automative_XAI class and run all explainers
    automative_xai = Automotive_XAI(
        model=model, features=features, X=X, y=y, feature_name=feature_name
    )
    automative_xai.do_PFI()
    automative_xai.do_ALE()
    automative_xai.do_LIME()
    automative_xai.do_IFI()
    automative_xai.do_SHAP()


test_dataset()
