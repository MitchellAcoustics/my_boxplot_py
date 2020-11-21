import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

#%% Summarise ---------------------------------------------------------------
def my_summary(dataset: pd.DataFrame, x: str, y: str, ci: float):
    df = dataset[[x, y]]
    ci_two_tailed = (1 - ci) / 2

    results = df.groupby(y).describe()
    results.columns = [col[1] for col in results.columns]
    results = results[["count", "mean", "std"]]

    results["se"] = results["std"] / np.sqrt(results["count"])

    results["min"] = (
        results["mean"]
        - stats.t.ppf(1 - ci_two_tailed, results["count"] - 1) * results["se"]
    )
    results["lower"] = results["mean"] - results["se"]
    results["upper"] = results["mean"] + results["se"]
    results["max"] = (
        results["mean"]
        + stats.t.ppf(1 - ci_two_tailed, results["count"] - 1) * results["se"]
    )

    return results


# Graph Settings


#%% Boxplots
def my_boxplot(
    data,
    x,
    y,
    xlab=x,
    ylab=y,
    width=0.8,
    fill="#2171b5",
    alpha=0.8,
    jitter_height=0.1,
    ci=0.95,
    points="dotplot",
    sort_by_mean=False,
    text_angle=0,
    text_size=12,
):
    summary = my_summary(data, x, y, ci=ci)
    if sort_by_mean:
        sorted_index = summary.sort_values("mean").index
        order = list(sorted_index)
    else:
        order = None

    # TODO: Still need to change all of these to plot what he's changed it to.
    sns.boxplot(y, x, data=data, showmeans=True, order=order)
    sns.stripplot(y, x, data=data, color="grey", alpha=alpha, order=order)
    plt.show()
    return print("plotting")


#%% testing

import pandas as pd
from sklearn import datasets

iris_obj = datasets.load_iris()
iris = pd.DataFrame(
    iris_obj.data,
    columns=iris_obj.feature_names,
    index=pd.Index([i for i in range(iris_obj.data.shape[0])]),
).join(
    pd.DataFrame(
        iris_obj.target,
        columns=pd.Index(["species"]),
        index=pd.Index([i for i in range(iris_obj.target.shape[0])]),
    )
)
iris.species.replace({0: "setosa", 1: "versicolor", 2: "virginica"}, inplace=True)

x = "sepal width (cm)"
y = "species"

summary = my_summary(iris, x, y, 0.95)
print(summary)

my_boxplot(iris, x, y, sort_by_mean=False)

# %%

# %%
