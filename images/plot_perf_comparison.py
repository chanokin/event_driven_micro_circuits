import numpy as np
from matplotlib import pyplot as plt
import h5py

columns = [
    "n_train",
    "n_validation",
    "n_hidden",
    "n_epochs",
    "run_time_seconds",
    "run_time_seconds_all",
    "run_time_training",
    "run_time_validation",
]

cols = {k: i for i, k in enumerate(columns)}

n_train_col = cols["n_train"]
n_hidden_col = cols["n_hidden"]
run_time_col = cols["run_time_seconds"]
run_time_val_col = cols["run_time_validation"]
run_time_train_col = cols["run_time_training"]

fnames = {
    "Rate change": "perf_drate_on_incr_hidden_layer.h5",
    "Fixed period": "perf_fixed_period_on_incr_hidden_layer.h5",
    "Continuous": "perf_std_on_incr_hidden_layer.h5",
}

colors = {
    "Rate change": "tab:blue",
    "Fixed period": "tab:orange",
    "Continuous": "tab:green",
}

ALL = "Full"
VAL = "Validation"
TRN = "Training"

MIN, AVG, MAX = range(3)

series = {}
for algorithm in fnames:
    fname = fnames[algorithm]
    with h5py.File(fname, "r") as h5f:
        vals = h5f['performance']
        print(h5f)

        all_n_train = vals[:, n_train_col]
        u_n_train = np.unique(all_n_train)
        # print(u_n_train)

        for n_train in u_n_train:
            n_train_dict = series.get(n_train, {})
            whr = np.where(vals[:, n_train_col] == n_train)
            vals_for_n_train = vals[whr]
            # print(vals_for_n_train.shape)

            u_n_hidden = np.unique(vals_for_n_train[:, n_hidden_col])
            n_hidden_vals = []
            run_time_all_vals = []
            run_time_val_vals = []
            run_time_train_vals = []

            min_run_time_all_vals = []
            min_run_time_val_vals = []
            min_run_time_train_vals = []

            max_run_time_all_vals = []
            max_run_time_val_vals = []
            max_run_time_train_vals = []
            for n_hidden in u_n_hidden:
                n_hidden_vals.append(int(n_hidden))
                n_hidden_rows = np.where(vals_for_n_train[:, n_hidden_col] == n_hidden)[0]

                avg_time = np.mean(vals_for_n_train[n_hidden_rows, run_time_col])
                min_time = np.min(vals_for_n_train[n_hidden_rows, run_time_col])
                max_time = np.max(vals_for_n_train[n_hidden_rows, run_time_col])
                run_time_all_vals.append(avg_time)
                min_run_time_all_vals.append(min_time)
                max_run_time_all_vals.append(max_time)

                avg_time = np.mean(vals_for_n_train[n_hidden_rows, run_time_val_col])
                min_time = np.min(vals_for_n_train[n_hidden_rows, run_time_val_col])
                max_time = np.max(vals_for_n_train[n_hidden_rows, run_time_val_col])
                run_time_val_vals.append(avg_time)
                min_run_time_val_vals.append(min_time)
                max_run_time_val_vals.append(max_time)

                avg_time = np.mean(vals_for_n_train[n_hidden_rows, run_time_train_col])
                min_time = np.min(vals_for_n_train[n_hidden_rows, run_time_train_col])
                max_time = np.max(vals_for_n_train[n_hidden_rows, run_time_train_col])
                run_time_train_vals.append(avg_time)
                min_run_time_train_vals.append(min_time)
                max_run_time_train_vals.append(max_time)

            all_dict = n_train_dict.get(ALL, {})
            all_dict[algorithm] = [
                n_hidden_vals,
                run_time_all_vals,
                min_run_time_all_vals,
                max_run_time_all_vals
            ]
            n_train_dict[ALL] = all_dict

            val_dict = n_train_dict.get(VAL, {})
            val_dict[algorithm] = [
                n_hidden_vals,
                run_time_val_vals,
                min_run_time_val_vals,
                max_run_time_val_vals,
            ]
            n_train_dict[VAL] = val_dict

            trn_dict = n_train_dict.get(TRN, {})
            trn_dict[algorithm] = [
                n_hidden_vals,
                run_time_train_vals,
                min_run_time_train_vals,
                max_run_time_train_vals,
            ]
            n_train_dict[TRN] = trn_dict

            series[n_train] = n_train_dict

plot_order = [VAL, TRN, ALL]
fw = 1.5
for n_train in series:
    fig, axs = plt.subplots(1, 3, figsize=(10 * fw, 3 * fw), sharey=True)
    plt.suptitle(f"{(int(n_train) + 1) * 3} input samples")

    for imeas, measure in enumerate(plot_order):
        ax = axs[imeas]
        ax.set_title(f"{measure}")

        for algorithm in series[n_train][measure]:
            x, y, min_y, max_y = series[n_train][measure][algorithm]
            ax.fill_between(x, min_y, max_y, alpha=0.3, color=colors[algorithm],
                            linewidth=0)
            ax.plot(x, y, label=algorithm, color=colors[algorithm])

        if imeas == 0:
            ax.set_ylabel("Run time [s]")

        ax.set_xlabel("Number of neurons")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"perf_comparison_with_{int(n_train)}_inputs.png", dpi=300)
plt.show()