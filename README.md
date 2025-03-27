# 🚀⚡ RankFusion - Simple, Quick, and Robust Rank Aggregation for Search & Recommendations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![GitHub stars](https://img.shields.io/github/stars/MSWinds/rankfusion?style=social)

This repository provides an implementation of various rank fusion algorithms commonly used in information retrieval and recommendation systems. These algorithms combine rankings from multiple sources (e.g., different search engines, recommendation models) to generate a unified, more accurate final ranking.

## 🚀 Features

### 🧠 Algorithms

* **Reciprocal Rank Fusion (RRF):** Combines ranks by summing their reciprocals.
* **Inverse Square Rank Fusion (ISR):** Similar to RRF but uses the inverse square of the ranks.
* **Relative Distance Fusion (RDF):** Normalizes scores before combining them, considering the relative distances between scores. Offers two normalization strategies:
  * **Min-Max:** Normalizes scores based on the minimum and maximum values in each ranking.
  * **Distribution-Based:** Normalizes scores based on the mean and standard deviation of each ranking.

### 🔧 Flexibility

* **Customizable weights:** Assign different weights to each ranking source to prioritize certain sources over others.
* **Ratio-Based Weights:** Now supports raw ratio weights (e.g., `[1,1,2]`), automatically normalizing them to sum to 1.
* **Parameter tuning:** Adjust parameters like `k` (for RRF and ISR) to fine-tune the fusion behavior.

### 🎯 Ease of Use

* **Intuitive API:** A simple `DistanceRankFusion` class handles the entire fusion process.
* **Clear documentation:** Well-documented code and examples for easy understanding and integration.

---

## 📊 How the Algorithms Work

### 1️⃣ Reciprocal Rank Fusion (RRF)

RRF assigns a score to each document based on its rank across multiple ranking sources:

$$
\text{RRF Score} = \sum_{j=1}^{N} \left(\frac{1}{\text{rank}_j + k} \times w_j\right)
$$

where:

- \( $$\text{rank}_j$$ is the rank of the document in the $$j$$-th source.
- $$k$$ is a parameter (default: 60) that prevents ranks from being too dominant. You can try higher k to have more even ranking.
- $$w_j$$ is the weight assigned to the $$j$$-th source.
- $$N$$ is the total number of sources.

A higher weight $$w_j$$ increases the influence of that ranking source.

### 2️⃣ Inverse Square Rank Fusion (ISR)

ISR modifies RRF by using the inverse square of the rank:

$$
\text{ISR Score} = \sum_{j=1}^{N} \left(\frac{1}{(\text{rank}_j + k)^2} \times w_j\right)
$$

### 3️⃣ Relative Distance Fusion (RDF)

RDF normalizes scores before combining them. There are two normalization strategies:

- #### 🅰️ **Min-Max Normalization**

Each score is scaled between 0 and 1 based on the min and max values:

$$
\text{normalized score} = \frac{\text{score} - \min(\text{score})}{\max(\text{score}) - \min(\text{score})}
$$

The final RDF score is:

$$
\text{RDF Score} = \sum_{j=1}^{N} (\text{normalized score}_j \times w_j)
$$

- #### 🅱️ **Distribution-Based Normalization** (RDFDB)

Scores are normalized based on the mean and standard deviation:

$$
\text{normalized score} = \frac{\text{score} - (\mu - 3\sigma)}{(\mu + 3\sigma) - (\mu - 3\sigma)}
$$

where $$\mu$$ is the mean and $$\sigma$$ is the standard deviation.  
The final RDF score is calculated the same way as in min-max normalization.

This method ensures better handling of outliers by scaling scores dynamically.

---

## 📥 Installation

You can install RankFusion via pip:

```bash
pip install rankfusion
```

---

## 📌 Usage Example

```python
import pandas as pd
from RankFusion import DistanceRankFusion

# Example dataset with ranking scores from different sources
data = {
    'algo1': [0.9, 0.7, 0.3, 0.1],
    'algo2': [0.8, 0.6, 0.4, 0.2],
    'algo3': [0.95, 0.65, 0.35, 0.15]
}
df = pd.DataFrame(data)

# Apply weighted Reciprocal Rank Fusion
fusion = DistanceRankFusion(
    df=df,
    score_columns=['algo1', 'algo2', 'algo3'],
    algorithm='rrf',
    weights=[1, 1, 2],  # Use raw ratio weights
    use_ratio_weights = True,  # Enables automatic normalization
    k=60,
    higher_is_better=True
)

result = fusion.get_result()
print(result)
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
