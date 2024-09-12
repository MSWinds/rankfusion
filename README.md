# RankFusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an implementation of various rank fusion algorithms commonly used in information retrieval and recommendation systems. These algorithms combine rankings from multiple sources (e.g., different search engines, recommendation models) to generate a unified, more accurate final ranking.

## Features

* **Algorithms:**
    * **Reciprocal Rank Fusion (RRF):** Combines ranks by summing their reciprocals.
    * **Inverse Square Rank Fusion (ISR):** Similar to RRF but uses the inverse square of the ranks.
    * **Relative Distance Fusion (RDF):** Normalizes scores before combining them, considering the relative distances between scores. Offers two normalization strategies:
        * **Min-Max:** Normalizes scores based on the minimum and maximum values in each ranking.
        * **Distribution-Based:** Normalizes scores based on the mean and standard deviation of each ranking.

* **Flexibility:**
    * **Customizable weights:** Assign different weights to each ranking source to prioritize certain sources over others.
    * **Parameter tuning:** Adjust parameters like `k` (for RRF and ISR) to fine-tune the fusion behavior.

* **Ease of use:**
    * **Intuitive API:** A simple `DistanceRankFusion` class handles the entire fusion process.
    * **Clear documentation:** Well-documented code and examples for easy understanding and integration.
