import json
import numpy as np
import os
import matplotlib.pyplot as plt
import sys


def get_the_result(param, file_dir):
    # dataset_name: 4_shot
    # file_dir: ./experiment
    results = {}
    datasets = os.listdir(file_dir)
    for dataset in datasets:
        print('\n\n\n ------------------')
        print(dataset)
        real_file_dir = os.path.join(file_dir, dataset)
        if dataset == "dbpedia":
            param = "1_shot"
        else:
            param = "4_shot"
        if dataset == "archive_cb":
            real_file_name = f"result_cb_{param}"
        else:
            real_file_name = f"result_{dataset}_{param}"
        

        # for each dataset
        result = get_different_models(real_file_dir, real_file_name)
        results[dataset] = result
    return results


def get_different_models(file_dir, file_name):
    # file_name: result_agnews_4_shot
    # file_dir: ./experiment/agnews

    def file_exist():
        for file in os.listdir(file_dir):
            if file.startswith(this_file_name):
                return True
        return False

    model_names = ["gpt2", "gpt-neo-125M", "gpt2-medium", "gpt2-large", "gpt-neo-1.3B", 
        "gpt2-xl", "gpt-neo-2.7B", "sharded-gpt-j-6B"]
    results = dict()
    for model_name in model_names:
        print('\n\n')
        print(model_name)
        this_file_name = f"{file_name}_{model_name}_seed"
        if file_exist():
            result = average_seed(file_dir, this_file_name)
            print(f"Acc: {result['acc_stats'][0]:.1%} +/- {result['acc_stats'][1]:.1%}")
            print(f"Ent: {result['entropys_stats'][0]:.1%} +/- {result['entropys_stats'][1]:.1%}")
            results[model_name] = result
    return results



def average_seed(file_dir, file_name):
    # file_name: result_agnews_4_shot_gpt2-large_seed
    # file_dir: ./experiment/agnews
    accs = []
    acc1 = []
    acc2 = []
    topk_acc1 = []
    topk_acc2 = []
    topk = 0
    entropies = []
    entropys1 = []
    entropys2 = []
    pearsonr_corr1 = []
    pearsonr_corr2 = []
    spearmanr_corr1 = []
    spearmanr_corr2 = []


    for i in range(1, 6):
        temp_file_name = f"{file_name}{i}.json"
        temp_file_dir = os.path.join(file_dir, temp_file_name)
        if not os.path.exists(temp_file_dir):
            continue
        
        result = get_one(temp_file_dir)

        accs.append(result['acc'])
        acc1.append(result['acc_stats'][0])
        acc2.append(result['acc_stats'][1])
        topk_acc1.append(result['topk_acc_stats'][0])
        topk_acc2.append(result['topk_acc_stats'][1])
        topk = result['topk']
        entropies.append(result['entropys'])
        entropys1.append(result['entropys_stats'][0])
        entropys2.append(result['entropys_stats'][1])
        pearsonr_corr1.append(result['pearsonr_corr'][0])
        pearsonr_corr2.append(result['pearsonr_corr'][1])
        spearmanr_corr1.append(result['spearmanr_corr'][0])
        spearmanr_corr2.append(result['spearmanr_corr'][1])
    
    results = {
        'acc': np.mean(np.array(accs), axis = 0).tolist(),
        'acc_stats': [np.mean(np.array(acc1)), np.mean(np.array(acc2))],
        'topk_acc_stats': [np.mean(np.array(topk_acc1)), np.mean(np.array(topk_acc2))],
        'topk': topk,
        'entropies': np.mean(np.array(entropies), axis = 0).tolist(),
        'entropys_stats': [np.mean(np.array(entropys1)), np.mean(np.array(entropys2))],
        'pearsonr_corr': [np.mean(np.array(pearsonr_corr1)), np.mean(np.array(pearsonr_corr2))],
        'spearmanr_corr': [np.mean(np.array(spearmanr_corr1)), np.mean(np.array(spearmanr_corr2))],
    }
    return results


def get_one(file_dir):
    f = open(file_dir, "r")
    data = json.load(f)
    # 'acc_stats', 'topk_acc_stats', 'topk', 'entropys', 'acc', 'ckpt', 'ckpt_gen', 'pearsonr_corr', 'spearmanr_corr'
    # data['acc_stats']: list of 2
    # data['topk_acc_stats']: list of 2
    # data['topk']: 4
    # data['entropys']: list of 24
    # data['acc']: list of 24 # each acc (we can ignore it)
    # data['ckpt']: data directory (true)
    # data['ckpt_gen']: data dir (fake)
    # data['pearsonr_corr']: list of 2
    # data['spearmanr_corr']: list of 2
    entropys = np.array(data['entropys'])
    entropy_result = [np.mean(entropys), np.std(entropys)]
    result = {
        'acc': data['acc'],
        'acc_stats': data['acc_stats'],
        'topk_acc_stats': data['topk_acc_stats'],
        'topk': data['topk'],
        'entropys': data['entropys'],
        'entropys_stats': entropy_result,
        'pearsonr_corr': data['pearsonr_corr'],
        'spearmanr_corr': data['spearmanr_corr']
    }
    f.close()
    return result






def plot1(dataset, result, save_dir, model_sizes):
    accs = []
    acc_mean = []
    acc_std = []
    topk_acc_mean = []
    topk_acc_std = []
    models = []
    for model in result:
        models.append(model)
        accs.append(result[model]['acc'])
        acc_mean.append(result[model]['acc_stats'][0])
        acc_std.append(result[model]['acc_stats'][1])
        topk_acc_mean.append(result[model]['topk_acc_stats'][0])
        topk_acc_std.append(result[model]['topk_acc_stats'][1])
    
    model_s_s = [model_sizes[i] for i in models]
    X = list(range(1, 1+len(acc_mean)))
    # plt.figure(figsize=(12, 7))
    plt.boxplot(accs)
    
    plt.xticks(X, model_s_s) #, rotation = 30)
    plt.xlabel('models')
    plt.ylabel('accuracies')
    plt.title(f'{dataset}: model accuracy wrt model types')
    plt.savefig(f"{save_dir}_model_accuracy_wrt_model_types.jpg")

    X = list(range(len(acc_mean)))
    plt.clf()
    plt.errorbar(X, acc_mean, yerr = acc_std, alpha=0.5, capsize=10, fmt='.', label = "accuracy")
    plt.errorbar(X, topk_acc_mean, yerr = topk_acc_std, alpha=0.5, capsize=10, fmt='.', label = "topk_accuracy")
    plt.xticks(X, model_s_s)
    plt.xlabel('models')
    plt.ylabel('accuracies')
    plt.title(f'{dataset}: model accuracies wrt model types')
    plt.legend()
    plt.savefig(f"{save_dir}_model_accuracies_wrt_model_types.jpg")



def plot2(dataset, result, save_dir, model_sizes):
    plt.clf()
    all_entropies = []
    entropy_mean = []
    entropy_std = []
    models = []
    for model in result:
        models.append(model)
        all_entropies.append(result[model]['entropies'])
        entropy_mean.append(result[model]['entropys_stats'][0])
        entropy_std.append(result[model]['entropys_stats'][1])

    X = list(range(24))
    colors = ['r', 'm', 'y', 'g', 'b', 'c', 'forestgreen', 'violet']
    for idx, i in enumerate(all_entropies):
        plt.scatter(X, i, c = colors[idx], label = model_sizes[models[idx]])
        plt.fill_between(X, [entropy_mean[idx] - entropy_std[idx]]*24, [entropy_mean[idx] + entropy_std[idx]]*24,
                 color=colors[idx], alpha=0.05)
    plt.xlabel("permutation")
    plt.ylabel("entropy")
    plt.title(f"{dataset}: entropy in different permutations")
    plt.legend()
    plt.savefig(f'{save_dir}_entropy_in_different_permutations.jpg')



def plot3(dataset, result, save_dir, model_sizes):
    plt.clf()
    pearsonr_corr_mean = []
    pearsonr_corr_std = []
    spearmanr_corr_mean = []
    spearmanr_corr_std = []
    models = []
    for model in result:
        models.append(model)
        pearsonr_corr_mean.append(result[model]['pearsonr_corr'][0])
        pearsonr_corr_std.append(result[model]['pearsonr_corr'][1])
        spearmanr_corr_mean.append(result[model]['spearmanr_corr'][0])
        spearmanr_corr_std.append(result[model]['spearmanr_corr'][1])
    
    model_s_s = [model_sizes[i] for i in models]
    X = list(range(len(pearsonr_corr_mean)))
    plt.errorbar(X, pearsonr_corr_mean, yerr = pearsonr_corr_std, alpha=0.5, capsize=10, fmt='.', label = "pearsonr_corr")
    plt.errorbar(X, spearmanr_corr_mean, yerr = spearmanr_corr_std, alpha=0.5, capsize=10, fmt='.', label = "spearmanr_corr")
    plt.xticks(X, model_s_s)
    plt.xlabel('models')
    plt.ylabel('pearsonr_corr and spearmanr_corr')
    plt.title(f'{dataset}: pearsonr_corr and spearmanr_corr wrt model types')
    plt.legend()
    plt.savefig(f"{save_dir}_pearsonr_corr_and_spearmanr_corr_wrt_model_types.jpg")



def plot_one_dataset(dataset, result):
    # acc_stats, topk_acc_stats one plot
    # entropies, entropys_stats dashline one plot
    # pearsonr_corr, spearmanr_corr one plot

    if dataset == "dbpedia":
        shot = "1_shot"
    else:
        shot = "4_shot"
    save_dir = f"./figure_result/{dataset}_{shot}"

    model_sizes = {
        "gpt2": "0.1B",
        "gpt-neo-125M": "0.1B",
        "gpt2-medium": "0.3B",
        "gpt2-large": "0.7B",
        "gpt-neo-1.3B": "1.3B",
        "gpt2-xl": "1.5B",
        "gpt-neo-2.7B": "2.7B",
        "sharded-gpt-j-6B": "6B"
    }

    # plot 1
    plot1(dataset, result, save_dir, model_sizes)

    # plot 2
    plot2(dataset, result, save_dir, model_sizes)

    # plot 3
    plot3(dataset, result, save_dir, model_sizes)



def main():
    param = "4_shot"
    file_dir = "./experiment"

    all_results = get_the_result(param, file_dir)
    for dataset in all_results:
        result = all_results[dataset]

        plot_one_dataset(dataset, result)
    

if __name__ == "__main__":
    main()
