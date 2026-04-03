from process.data.sample import load_sample_data
from process.model.utils import check_deps_charts, obtain_all_tasks
from os.path import exists as os_path_exists
from os import makedirs as os_makedirs
from process.model.stochastic_impute import stochastic_impute
from process.postp.vis import plot_distribution
from yaml import safe_load

output_dir = "./output"

if not os_path_exists(output_dir):
    os_makedirs(output_dir)

data = load_sample_data(refresh=True)

with open("config.yml", "r") as fid:
    cfg = safe_load(fid)

proc_data = data["seed"].copy()
task_list = obtain_all_tasks(cfg["tasks"], cfg["cfg"])

syn_pop = stochastic_impute(proc_data, data, task_list, output_dir=output_dir)


check_deps_charts(cfg["cfg"], output_dir=output_dir)
plot_distribution(syn_pop, ["age"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["work_hours"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["travel_to_work"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["occupation"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["occupation", "income"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["age", "work_hours"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["gender", "work_hours"], output_dir=output_dir, dropna=True)
plot_distribution(syn_pop, ["industry", "income"], output_dir=output_dir, dropna=True)
