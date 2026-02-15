from pandas import Series as pdSeries
from pandas import DataFrame as pdDataFrame
from sklearn.preprocessing import LabelEncoder
from graphviz import Digraph


def create_input_data(input_data: pdDataFrame, deps_cols: list, data_range: dict):
    # build a combined boolean mask
    mask = pdSeries(True, index=input_data.index)

    for col in deps_cols:
        vals = data_range.get(col, [])
        mask &= input_data[col].isin(vals)

    # split into two dataframes
    df1 = input_data[mask]
    df2 = input_data[~mask]

    return {"data_in_range": df1[deps_cols], "data_out_range": df2[deps_cols]}


def check_deps_charts(models_cfg: dict, output_dir: str = "./output"):
    # Create directed graph
    dot = Digraph(format="png")
    dot.attr(rankdir="LR", size="8,5")

    added_edges = set()

    # Add edges: features → targets
    for model_name, m_cfg in models_cfg.items():
        for feature in m_cfg["features"]:
            for target in m_cfg["targets"]:
                edge = (feature, target)
                if edge not in added_edges:
                    dot.edge(feature, target, label="")
                    added_edges.add(edge)

    # Save and render
    # dot.render("model_flow", view=True)
    output_path = dot.render(f"{output_dir}/model_flow", format="png", cleanup=True)
    print(f"Flow chart saved to: {output_path}")


def obtain_all_tasks(tasks_cfg: dict):
    try:
        all_tasks = tasks_cfg.split("->")
    except AttributeError:
        all_tasks = [tasks_cfg]

    return all_tasks


def obtain_cur_task(proc_task: str):
    proc_task = proc_task.strip()

    if proc_task.startswith("[") and proc_task.endswith("]"):
        proc_task = [item.strip() for item in proc_task.strip("[]").split(", ")]
    else:
        proc_task = [proc_task]

    return proc_task
