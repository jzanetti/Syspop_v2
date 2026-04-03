from yaml import safe_load
from process.data.data_wrapper import obtain_data_wrapper
from process.model.utils import check_deps_charts, obtain_all_tasks
from os.path import exists as os_path_exists
from os import makedirs as os_makedirs
from process.model.stochastic_impute import stochastic_impute
from process.postp.vis import plot_distribution

with open("config.yml", "r") as fid:
    cfg = safe_load(fid)

output_dir = "./output"

if not os_path_exists(output_dir):
    os_makedirs(output_dir)

data = obtain_data_wrapper(
    cfg["tables"],
    data_types=[
        "pop",
        "industry",
        "occupation",
        "occupation_income",
        "industry_income",
        "travel_to_work",
        "work_hours",
    ],
)

check_deps_charts(cfg["models"]["cfg"], output_dir=output_dir)

proc_data = data["pop"].copy()
task_list = obtain_all_tasks(cfg["models"]["tasks"], cfg["models"]["cfg"])

syn_pop = stochastic_impute(proc_data, data, task_list, output_dir=output_dir)


plot_distribution(syn_pop, ["age"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["occupation"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["occupation", "income"], output_dir=output_dir, dropna=True)
