from yaml import safe_load
from process.data_wrapper import obtain_data_wrapper
from process.model_wrapper import run_model_train_wrapper, run_model_pred_wrapper
from process.model.utils import check_deps_charts, obtain_all_tasks, obtain_cur_task
from process.data.utils import check_data_consistency
from os.path import exists as os_path_exists
from os import makedirs as os_makedirs

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
check_data_consistency(data, output_dir=output_dir)

for proc_task in obtain_all_tasks(cfg["models"]["tasks"]):

    proc_task = obtain_cur_task(proc_task)

    run_model_train_wrapper(data, cfg["models"]["cfg"], data_types=proc_task)
    run_model_pred_wrapper(
        data,
        data_types=proc_task,
        postp_algorithm=cfg["models"].get("postp_algorithm", None),
        output_dir=output_dir,
    )
