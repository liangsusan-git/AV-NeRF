import os
import json
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best', action="store_true")
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    env_dict = {"office": list(range(1, 6)),
               "house": list(range(6, 9)),
               "apartment": list(range(9, 12)),
               "outdoor": list(range(12, 14))}

    print("===================================")
    result = []
    avg_dict = {}
    for env_name, env_id in env_dict.items():
        loss_list = []
        for i in env_id:
            log_path = os.path.join(args.log_dir, f"{i}/audio_output/", args.output_dir, "eval_loss.pkl")
            try:
                loss = pickle.load(open(log_path, "rb"))
                if args.best:
                    loss = sorted(loss, key=lambda x: (x["env"], x["mag"]))[0]
                else:
                    loss = loss[-1]
                loss_list.append(loss)
            except:
                print(f"{log_path} Not exists")
        if len(loss_list) == 0: continue

        loss_dict = {k: [] for k in loss_list[0].keys()}
        for item in loss_list:
            for k in item.keys():
                loss_dict[k].append(item[k])
        print(env_name, end=" ")
        for k in loss_dict.keys():
            loss_dict[k] = sum(loss_dict[k]) / len(loss_dict[k])
            print(k, round(loss_dict[k], 3), end=" ")
            if k == "mag" or k == "env": result.append(round(loss_dict[k], 3))
            if not k in avg_dict.keys():
                avg_dict[k] = []
            avg_dict[k].append(loss_dict[k])
        print()

    print("average", end=" ")
    for k in avg_dict.keys():
        avg_dict[k] = sum(avg_dict[k]) / len(avg_dict[k])
        print(k, round(avg_dict[k], 3), end=" ")
        if k == "mag" or k == "env": result.append(round(avg_dict[k], 3))
    print()
    result = [f"{item:.3f}" for item in result]
    result_copy = [item for item in result]
    result[::2] = result_copy[1::2]
    result[1::2] = result_copy[::2]
    print(" & ".join(result))
    print("===================================")