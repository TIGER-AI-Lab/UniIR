import argparse
import yaml
from copy import deepcopy

dataset_to_query_file_middle_name_map = {
    "visualnews": ["visualnews_task0", "visualnews_task3"],
    "mscoco": ["mscoco_task0", "mscoco_task3"],
    "fashion200k": ["fashion200k_task0", "fashion200k_task3"],
    "webqa": ["webqa_task1", "webqa_task2"],
    "edis": ["edis_task2"],
    "nights": ["nights_task4"],
    "oven": ["oven_task6", "oven_task8"],
    "infoseek": ["infoseek_task6", "infoseek_task8"],
    "fashioniq": ["fashioniq_task7"],
    "cirr": ["cirr_task7"],
}

dataset_to_metric_map = {
    "visualnews": ["Recall@1, Recall@5, Recall@10"] * 2,
    "mscoco": ["Recall@1, Recall@5, Recall@10"] * 2,
    "fashion200k": ["Recall@10, Recall@20, Recall@50"] * 2,
    "webqa": ["Recall@1, Recall@5, Recall@10"] * 2,
    "edis": ["Recall@1, Recall@5, Recall@10"],
    "nights": ["Recall@1, Recall@5, Recall@10"],
    "oven": ["Recall@1, Recall@5, Recall@10"] * 2,
    "infoseek": ["Recall@1, Recall@5, Recall@10"] * 2,
    "fashioniq": ["Recall@10, Recall@20, Recall@50"],
    "cirr": ["Recall@1, Recall@5, Recall@10"],
}


def load_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def print_yaml(data):
    print(yaml.dump(data, default_flow_style=False, sort_keys=False, indent=4))


def update_mbeir_yaml_instruct_status(yaml_file_path, enable_instruct):
    print(f"Updating YAML {yaml_file_path} for instruct status: {enable_instruct}")
    yaml_data = load_yaml(yaml_file_path)
    if enable_instruct:
        yaml_data["experiment"]["instruct_status"] = "Instruct"
        if "data_config" in yaml_data:
            yaml_data["data_config"]["enable_query_instruct"] = True
            yaml_data["data_config"]["enable_cand_instruct"] = True
        else:
            print(f"YAML {yaml_file_path} does not have data_config.")
    else:
        yaml_data["experiment"]["instruct_status"] = "NoInstruct"
        if "data_config" in yaml_data:
            yaml_data["data_config"]["enable_query_instruct"] = False
            yaml_data["data_config"]["enable_cand_instruct"] = False
        else:
            print(f"YAML {yaml_file_path} does not have data_config.")
    print(f"Updated YAML {yaml_file_path} for instruct status:{enable_instruct} :")
    print_yaml(yaml_data)
    save_yaml(yaml_data, yaml_file_path)


def update_mbeir_config_dir_instruct_status(config_dir, enable_instruct):
    print(f"Updating config dir {config_dir} for instruct status: {enable_instruct}")

    # Update the Embed YAML file
    embed_yaml_file_path = f"{config_dir}/embed.yaml"
    update_mbeir_yaml_instruct_status(embed_yaml_file_path, enable_instruct)

    # Update the Index YAML file
    index_yaml_file_path = f"{config_dir}/index.yaml"
    update_mbeir_yaml_instruct_status(index_yaml_file_path, enable_instruct)

    # Update the Retrieval YAML file
    retrieval_yaml_file_path = f"{config_dir}/retrieval.yaml"
    update_mbeir_yaml_instruct_status(retrieval_yaml_file_path, enable_instruct)


def update_mbeir_eval_config_dir_for_single_data_exp(config_dir, dataset_name, enable_instruct):
    print(f"Updating config dir {config_dir} for dataset {dataset_name} and instruct status: {enable_instruct}")

    update_mbeir_config_dir_instruct_status(config_dir, enable_instruct)

    # Step 1. Update the Embed YAML file
    embded_yaml_file_path = f"{config_dir}/embed.yaml"

    # Update dataset
    embed_yaml = load_yaml(embded_yaml_file_path)
    embed_yaml["experiment"]["dataset_name"] = dataset_name
    query_file_list = dataset_to_query_file_middle_name_map[dataset_name]

    # Update datasets_name and correspond_cand_pools_name under test_datasets_config
    embed_yaml["embed_config"]["test_datasets_config"]["enable_embed"] = True
    test_datasets_config_datasets_name_to_embed = []
    test_datasets_config_correspond_cand_pools_name_to_embed = []
    cand_pools_name_to_embed = []
    for query_file_name in query_file_list:
        test_datasets_config_datasets_name_to_embed.append(query_file_name)
        if "mscoco" in query_file_name:
            cand_pools_name_to_embed.append(f"{query_file_name}_test")  # MSCOCO has split specific cand pools
            test_datasets_config_correspond_cand_pools_name_to_embed.append(f"{query_file_name}_test")
        else:
            cand_pools_name_to_embed.append(query_file_name)
            test_datasets_config_correspond_cand_pools_name_to_embed.append(query_file_name)
    embed_yaml["embed_config"]["test_datasets_config"]["datasets_name"] = test_datasets_config_datasets_name_to_embed
    embed_yaml["embed_config"]["test_datasets_config"][
        "correspond_cand_pools_name"
    ] = test_datasets_config_correspond_cand_pools_name_to_embed
    embed_yaml["embed_config"]["cand_pools_config"]["cand_pools_name_to_embed"] = cand_pools_name_to_embed
    embed_yaml["embed_config"]["cand_pools_config"]["embed_union_pool"] = False

    # Step 2.  Update the Index YAML file
    index_yaml_file_path = f"{config_dir}/index.yaml"

    # Update dataset
    index_yaml = load_yaml(index_yaml_file_path)
    index_yaml["experiment"]["dataset_name"] = dataset_name
    index_yaml["index_config"]["cand_pools_config"]["cand_pools_name_to_idx"] = deepcopy(
        embed_yaml["embed_config"]["cand_pools_config"]["cand_pools_name_to_embed"]
    )

    # Step 3. Update the Retrieval YAML file
    retrieval_yaml_file_path = f"{config_dir}/retrieval.yaml"

    # Update dataset
    retrieval_yaml = load_yaml(retrieval_yaml_file_path)
    retrieval_yaml["experiment"]["dataset_name"] = dataset_name

    # Update the test config
    retrieval_yaml["retrieval_config"]["test_datasets_config"]["enable_retrieve"] = True
    retrieval_yaml["retrieval_config"]["test_datasets_config"]["datasets_name"] = deepcopy(
        embed_yaml["embed_config"]["test_datasets_config"]["datasets_name"]
    )
    retrieval_yaml["retrieval_config"]["test_datasets_config"]["correspond_cand_pools_name"] = deepcopy(
        embed_yaml["embed_config"]["test_datasets_config"]["correspond_cand_pools_name"]
    )
    retrieval_yaml["retrieval_config"]["test_datasets_config"]["correspond_qrels_name"] = deepcopy(
        embed_yaml["embed_config"]["test_datasets_config"]["datasets_name"]
    )
    retrieval_yaml["retrieval_config"]["test_datasets_config"]["correspond_metrics_name"] = deepcopy(
        dataset_to_metric_map[dataset_name]
    )

    # Pretty print the updated config
    print("Updated Embed YAML:")
    print_yaml(embed_yaml)
    save_yaml(embed_yaml, embded_yaml_file_path)
    print(f"Updated {embded_yaml_file_path} for dataset {dataset_name}")

    print("Updated Index YAML:")
    print_yaml(index_yaml)
    save_yaml(index_yaml, index_yaml_file_path)
    print(f"Updated {index_yaml_file_path} for dataset {dataset_name}")

    print("Updated Retrieval YAML:")
    print_yaml(retrieval_yaml)
    save_yaml(retrieval_yaml, retrieval_yaml_file_path)
    print(f"Updated {retrieval_yaml_file_path} for dataset {dataset_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Utility functions for experiment configurations.")
    parser.add_argument(
        "--update_mbeir_yaml_instruct_status",
        action="store_true",
        help="Update instruct status for mbeir yamls",
    )
    parser.add_argument("--mbeir_yaml_file_path", type=str, default="ReplaceMe")

    # Args for single data fine-tuning experiment
    parser.add_argument(
        "--update_mbeir_eval_config_dir_for_single_data_exp",
        action="store_true",
        help="Update YAMLs for single data fine-tuning experiment",
    )
    parser.add_argument(
        "--single_data_exp_eval_config_dir",
        type=str,
        default="ReplaceMe",
        help="Dir Path to the YAML configuration file",
    )
    parser.add_argument(
        "--update_mbeir_train_yaml_for_single_data_exp",
        action="store_true",
        help="Update YAMLs for single data fine-tuning experiment",
    )
    parser.add_argument(
        "--single_data_exp_train_config_file_path",
        type=str,
        default="ReplaceMe",
        help="Update YAMLs for single data fine-tuning experiment",
    )
    parser.add_argument(
        "--single_data_exp_dataset",
        type=str,
        default="ReplaceMe",
        help="Dataset name for single data fine-tuning experiment",
    )

    parser.add_argument(
        "--enable_instruct",
        required=True,
        choices=["True", "False"],
        help="Whether to enable instruct or not. Required input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    enable_instruct = args.enable_instruct == "True"

    # Experiment: Single dataset fine-tuning
    if args.update_mbeir_train_yaml_for_single_data_exp:
        update_mbeir_yaml_instruct_status(args.single_data_exp_train_config_file_path, enable_instruct)
        yaml_data = load_yaml(args.single_data_exp_train_config_file_path)
        yaml_data["experiment"]["dataset_name"] = args.single_data_exp_dataset
        print(f"Updating YAML {args.single_data_exp_train_config_file_path} for dataset {args.single_data_exp_dataset}")
        print_yaml(yaml_data)
        save_yaml(yaml_data, args.single_data_exp_train_config_file_path)
    if args.update_mbeir_eval_config_dir_for_single_data_exp:
        update_mbeir_eval_config_dir_for_single_data_exp(
            args.single_data_exp_eval_config_dir,
            args.single_data_exp_dataset,
            enable_instruct,
        )

    # General update mbeir yaml instruct status
    if args.update_mbeir_yaml_instruct_status:
        if args.mbeir_yaml_file_path == "ReplaceMe":
            print("The default YAML file path has not been replaced with an actual file path.")
        update_mbeir_yaml_instruct_status(args.mbeir_yaml_file_path, enable_instruct)
