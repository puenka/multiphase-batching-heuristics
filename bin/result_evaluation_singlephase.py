# %%
import pathlib
import json
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = (12, 8)


# Load and prepare data
instance_dir = pathlib.Path("../data/instances-singlephase")
data_raw = []
for file in instance_dir.glob("*/statistics*"):
    new_data = json.loads(file.read_text())
    new_data["instance"] = file.parent.name
    data_raw.append(new_data)

data = (
    pl.DataFrame(data_raw)
    .filter(
        ~pl.col("algorithm").str.starts_with("zga"),  # remove ZGA from analysis
        ~pl.col("instance").str.starts_with(
            "debug"
        ),  # remove debug instances from data
    )
    .unnest(separator="::")  # unnest all structs (stats data with min/med/mean/max)
)
data = data.with_columns(
    base_algo=pl.col("algorithm").str.strip_suffix("-cvrp"),
    cvrp=pl.col("algorithm").str.ends_with("-cvrp"),
    instance_type=pl.col("instance").str.strip_chars_end("-01234"),
    order_length_ratio=pl.col("batched_order_length_stats::mean")
    / pl.col("instance_order_length_stats::mean"),
)
# data = data.filter(
#     pl.col("base_algo").is_in(["dga", "dgapq", "rdga-sqrt", "rdga-50", "neighbour-aisle"])
# )  # filter relevant algorithms for the paper

# data.write_csv("../data/publication-data/compiled-data-polars.csv")

summary_data = (
    data.group_by(["instance_type", "base_algo", "cvrp"])
    .agg(
        pl.selectors.numeric().mean(),
    )
    .with_columns(
        relative_objective_value=pl.col("objective_value")
        / pl.col("objective_value").min().over("instance_type", "cvrp"),
        relative_time_elapsed=pl.col("time_elapsed")
        / pl.col("time_elapsed")
        .get(pl.col("objective_value").arg_min())
        .over("instance_type"),
    )
)

# %% [markdown]
# Overview of computed instances

# We run the following parameters:

# - `small`, `medium`, `large` as specified by Zalando
# - `-fewzones` uses 10 zones, `-fewerzones` only 3 zones, with the same number of articles/items. Thus, there are many more items in each warehouse zone than in the reference instance of the same size.
# - `-longbatch` has 250 order per batch instead of 50
# - `-highratio` uses 15 items per article instead of 3

# Now, we check the RSD for the runs of a given parameter configuration (instance type, algo, CVRP solution method).

# As the RSD only significantly exceeds 10% for the picklist lengths statistics for algorithms where we cannot necessarily expect consistent outcomes across instances, we continue with the analysis on all data. See output below.

# %%
# Check RSD for all data for statistical consistency.

rsd = (
    data.select(pl.all().exclude("^.*(min|max)$"))
    .group_by(["instance_type", "base_algo", "cvrp"])
    .agg(pl.selectors.numeric().std() / pl.selectors.numeric().mean())
    .filter(pl.any_horizontal(pl.selectors.numeric() > 0.1))
)
for row in rsd.iter_rows(named=True):
    print(row["instance_type"], row["base_algo"], row["cvrp"])
    for name, val in row.items():
        if isinstance(val, float) and val > 0.1:
            print("\t", name, val)

# %% [markdown]

# Next, we present some baseline data about the runs, such as the objective function value and picklist lengths. Each of these datapoints is averaged over the 5 runs of the respective instance with the given algorithm.

# %%

ax = sns.scatterplot(
    summary_data.sort("instance_type"),
    x="instance_type",
    y="relative_objective_value",
    hue="base_algo",
    style="cvrp",
)
ax.set_title("Relative objective value (compared to min)")
plt.xticks(rotation=75)
# %%

ax = sns.scatterplot(
    summary_data.sort("instance_type"),
    x="instance_type",
    y="relative_time_elapsed",
    hue="base_algo",
    style="cvrp",
)
ax.set_title("Relative time (compared to best)")
plt.yscale("log")
plt.xticks(rotation=75)
# %% [markdown]
# # CVRP analysis

# We compare the algorithms with and without the CVRP solver. We present the improvement in objective value for each instance and algorithm and relate it to the increase in computation time. Note that a 10s time limit has been set for the CVRP solver in each batch.
# %%
cvrp_ratios = (
    summary_data.pivot("cvrp", index=["base_algo", "instance_type"])
    .with_columns(
        obj_ratio_percent=(
            100 * pl.col("objective_value_false") / pl.col("objective_value_true") - 100
        ).round(2),
        time_diff=(pl.col("time_elapsed_true") - pl.col("time_elapsed_false")).round(0),
    )
    .with_columns(
        (
            pl.selectors.contains("time_elapsed")
            | pl.selectors.contains("objective_value")
        ).round(),
    )
)

# cvrp_data_rounded = cvrp_data.with_columns(
#     pl.col("base_algo"), pl.selectors.float().round()
# )


# cvrp_ratios.pivot("instance_type", index="base_algo").select(
#     pl.all().exclude("base_algo")
# ).describe()

# %%
ax = sns.heatmap(
    cvrp_ratios.sort("instance_type")
    .pivot(on="instance_type", index="base_algo", values="obj_ratio_percent")
    .to_pandas()
    .set_index("base_algo"),
    annot=True,
)
ax.set_title("Relative performance of heuristic vs. CVRP")
plt.xticks(rotation=75)
# %%

ax = sns.heatmap(
    cvrp_ratios.sort("instance_type")
    .pivot(on="instance_type", index="base_algo", values="time_diff")
    .to_pandas()
    .set_index("base_algo"),
    annot=True,
)
ax.set_title("Runtime loss for CVRP solver vs. heuristic")
plt.xticks(rotation=75)
# %% [markdown]

# The data above shows that for the old instances, the CVRP solver had negligible performance gains. In the new parameters `few[er]zones`, however, we observe at least high single-digit percentage point gains. We can explain this by inspecting the picklist statistics for the different instances.

# %%

ax = sns.scatterplot(
    summary_data.sort("instance_type"),
    x="instance_type",
    y="picklist_length_stats::mean",
    hue="base_algo",
    style="cvrp",
)
ax.set_title("Mean picklist length")
plt.xticks(rotation=75)
# %% [markdown]
# For reference, we provide the median load of the container in each picklist. Compared to before, we achieve much higher loads now.
ax = sns.scatterplot(
    summary_data.sort("instance_type"),
    x="instance_type",
    y="picklist_container_level_stats::med",
    hue="base_algo",
    style="cvrp",
)
ax.set_title("Median container level")
plt.xticks(rotation=75)
# %% [markdown]

# To provide some better insight, we now restrict the analysis to the CVRP methods and observe the trend in the solution quality across the different algorithms again.

# %%

ax = sns.lineplot(
    summary_data.sort("instance_type").filter(~pl.col("cvrp")),
    x="instance_type",
    y="relative_objective_value",
    hue="base_algo",
)
ax.set_title("Relative objective value (compared to min)")
plt.xticks(rotation=75)

# %%
for group_name, group_data in (
    summary_data.sort("instance_type")
    .filter(~pl.col("cvrp"))
    .select(
        "instance_type",
        "base_algo",
        "relative_objective_value",
        "objective_value",
        "time_elapsed",
    )
    .group_by("instance_type")
):
    print("Instance:", group_name[0])
    print(group_data.select(pl.all().exclude("instance_type")))

# %% [markdown]
# # Benchmarks for the single phase
# We now exclude data including the CVRP routing and the reference DGA to get DGAPQ as the new reference

# data = data.filter(~pl.col("cvrp") & pl.col("base_algo") != "dga").select(pl.all().exclude("cvrp"))
data = data.filter(~pl.col("cvrp")).select(pl.all().exclude("cvrp"))


# # %%
# medium_stats = (
#     data.filter(pl.col("instance_type") == "medium")
#     .select(["algorithm", "instance", "time_elapsed", "objective_value"])
#     .pivot(on="algorithm", index="instance")
#     .describe()
# )
# large_stats = (
#     data.filter(pl.col("instance_type") == "large")
#     .select(["algorithm", "instance", "time_elapsed", "objective_value"])
#     .pivot(on="algorithm", index="instance")
#     .describe()
# )


# # %%

# summary_data = (
#     data.select(
#         [
#             "algorithm",
#             "instance_type",
#             "time_elapsed",
#             "objective_value",
#             "order_length_ratio",
#             pl.col("^.*::.*$"),  # all unnested structs
#             "avg_item_cost",
#         ]
#     )
#     .group_by("algorithm", "instance_type")
#     .mean()
#     .with_columns(
#         obj_ratio=pl.col("objective_value")
#         / pl.col("objective_value").min().over("instance_type"),
#         opt_elapsed_ratio=pl.col("time_elapsed")
#         / pl.col("time_elapsed")
#         .get(pl.col("objective_value").arg_min())
#         .over("instance_type"),
#     )
#     .sort("instance_type", "algorithm", descending=[True, False])
#     .with_columns(
#         pl.col("time_elapsed").round(0),
#         pl.col("objective_value").round(0),
#         pl.col("order_length_ratio").round(3),
#         pl.col("obj_ratio").round(3),
#         pl.col("opt_elapsed_ratio").round(5),
#     )
# )

# # %% [markdown]
# # # Investigate the order lengths


# # %%
