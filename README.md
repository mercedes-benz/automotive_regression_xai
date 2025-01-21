# automotive_regression_xai

<!-- SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH. -->
<!-- SPDX-License-Identifier: MIT -->

<!-- TABLE OF CONTENTS -->
<h2>Table of Contents</h2>
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#structure-of-the-repository">Structure of the Repository</a></li>
  <li><a href="#package-installation">Package Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#contact">Contact</a></li>
</ol>

## Provider Information

<!-- Disclaimer -->
Source code has been tested solely for our own use cases, which might differ from yours.
This project is actively maintained and contributing is endorsed.

<!-- ABOUT THE PROJECT -->
## About The Project
The `automotive_regression_xa` Python package has been developed as a reference to demonstrate explainable AI functions and operations, providing deeper insights into their behavior and decision-making processes. It showcases how these techniques can be utilized to support AI applications within the automotive industry.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure of the repository
```plaintext
├── automotive_regression_xai/
├── src/
│   └── xai_classes/
│       ├── xai_ale.py
│       ├── xai_ifi.py
│       ├── xai_lime.py
│       ├── xai_pfi.py
│       ├── xai_shap.py
│   └── automotive_regression_xai.py
│   └── sample_dataset.py
│   └── test.py

```
   <p align="right">(<a href="#readme-top">back to top</a>)</p>

## Package installation
1. **Clone the source code:**
   ```bash
   git clone <repository_url>
   cd automotive_regression_xai
   ```

2. **Install Python dependencies in your Python environment:**
   ```bash
   python3 -m pip install -e .
   ```

3. **Build a package and install it on the system:**
   ```bash
   python3 -m pip install --upgrade build
   python3 -m build
   pip install dist/*.tar.gz
   ```

4. **Verify if the installation was successful:**
   ```bash
   pip list | grep automotive_regression_xai
   ```

   <p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

#### ALE (Accumulated Local Effects):
Analyzes feature impacts by showing average changes in predictions as features vary, considering interactions while avoiding collinearity issues.

#### PFI (Permutation Feature Importance):
Measures feature importance by permuting feature values and observing changes in model performance.

#### IFI (Integrated Feature Importance):
Combines multiple importance measures for more holistic feature evaluation.

#### LIME (Local Interpretable Model-agnostic Explanations):
Explains individual predictions by approximating the model locally with interpretable surrogates like linear models.

#### SHAP (SHapley Additive exPlanations):
Assigns feature importance based on cooperative game theory, ensuring consistency and interpretability across all predictions.

### Example

Here's an example of how you might use these arguments:

- **Example 1:**

   ```bash
   cd src
   python test.py
   ```

   <p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing
The instructions on how to contribute can be found in the file [CONTRIBUTING.md](./CONTRIBUTING.md) in this repository.
   <p align="right">(<a href="#readme-top">back to top</a>)</p>

## License
The code is published under the MIT license. Further information on that can be found in the [LICENSE.md](./LICENSE.md) file in this repository.
   <p align="right">(<a href="#readme-top">back to top</a>)</p>
