# imports
from autograd import grad
import autograd.numpy as np_autograd
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
import time

# define plot font sizes globally
matplotlib.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


# given by professor
def generate_rreg_data(N, D, seed):
    rng = np.random.default_rng(seed)
    # generate multivariate t covariates with 10 degrees
    # of freedom and non-diagonal covariance
    t_dof = 10
    locs = np.arange(D).reshape((D, 1))
    cov = (t_dof - 2) / t_dof * np.exp(-((locs - locs.T) ** 2) / 4)
    Z = rng.multivariate_normal(np.zeros(D), cov, size=N)
    Z *= np.sqrt(t_dof / rng.chisquare(t_dof, size=(N, 1)))
    # generate responses using regression coefficients beta = (1, 2, ..., D)
    # and t-distributed noise
    true_beta = np.squeeze(np.log(1 + D) / (1 + locs))
    Y = Z.dot(true_beta) + rng.standard_t(t_dof, size=N)
    # for simplicity, center responses
    Y = Y - np.mean(Y)
    return Y, Z


# initializes the common parameters
def initialize_parameters(dataset):

    if dataset == "given":
        N = 10000  # number of observations
        D = 20  # size of feature space
        M = 50  # epochs
        B = 10  # batch size
        K = int(M * N / B)  # number of iterations #NOTE M x N = K * B
        eta = 0.5  # learning rate
        eta_decay = 25  # learning rate for polynomial decay
        alphas = [0.55, 0.75, 0.95]  # polynomial decay rate
        rng = np.random.default_rng(54)  # sets random number generator
        y, Z = generate_rreg_data(N, D, 0)
        nu = 5  # t_dof
        initial_values = [0, 5, 10, 20, 30, 40, 50]  # initial values of x

    elif dataset == "gpu":

        # cleans dataset
        gpu_dataset = pd.read_csv("./gpu_dataset/gpu_performance.csv")
        gpu_dataset["RUN_AVG_MS"] = np.log(
            gpu_dataset[["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]].mean(
                axis=1
            )
        )
        gpu_dataset.drop(
            labels=["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"],
            axis=1,
            inplace=True,
        )

        N = 50000  # number of observations
        D = len(gpu_dataset.iloc[0]) - 1  # size of feature space
        M = 50  # epochs
        B = 50  # batch size
        K = int(M * N / B)  # number of iterations #NOTE M x N = K * B
        eta = 0.5  # learning rate
        eta_decay = 25  # learning rate for polynomial decay
        alphas = [0.55, 0.75, 0.95]  # polynomial decay rate
        rng = np.random.default_rng(54)  # sets random number generator
        Z = gpu_dataset.drop("RUN_AVG_MS", axis=1).to_numpy()
        y = gpu_dataset["RUN_AVG_MS"].to_numpy()
        ind = rng.choice(len(Z), size=50000, replace=False)
        Z = Z[ind]
        y = y[ind]
        nu = 5  # t_dof
        initial_values = [0, 5, 10, 20, 30, 40, 50]  # initial values of x

    print("Observations (N):", N)
    print("Epochs (M):", M)
    print("Batch Size (B):", B)
    print("Iterations (K):", K)
    print("Learning Rate (eta):", eta)
    print("Learning Rate for Polynomial Decay (eta_decay):", eta_decay)
    print("Feature Space (D):", D)
    print("Degrees of Freedom (nu):", nu)

    return N, M, B, K, eta, eta_decay, alphas, rng, D, y, Z, nu, initial_values


# calculates the gradient manually
def calculate_manual_gradient(x, y, Z, nu, N, B):

    # initializes psi and beta
    psi = x[0]
    beta = x[1:]

    grad = np.zeros(len(beta) + 1)

    # # aggregates total loss across observations
    for n in range(B):
        z_n = Z[n]
        residual = y[n] - np.dot(z_n.T, beta)

        # calculates pder with respect to psi
        numerator_psi = (nu + 1) * np.exp(-2 * psi) * (residual**2)
        denominator_psi = nu + np.exp(-2 * psi) * (residual**2)
        grad[0] += 1 - (numerator_psi / denominator_psi)

        # calculates pder with respect to B_k
        for k in range(len(beta)):
            m_k = np.zeros(len(z_n))
            m_k[k] = 1
            numerator_beta_k = (
                (nu + 1) * np.exp(-2 * psi) * (np.dot(z_n.T, m_k)) * (residual)
            )
            denominator_beta_k = nu + np.exp(-2 * psi) * (residual**2)
            grad[k + 1] -= numerator_beta_k / denominator_beta_k

    # normalizes gradient vector by number of observations
    grad /= B

    # adds regularization term
    x = np.concatenate([[(psi)], beta])
    l2_term = x / N
    grad += l2_term

    return grad


# implements the loss function when nu < infinity
def loss_function(x, y, Z, nu, N):

    # initializes psi and beta
    psi = x[0]
    beta = x[1:]

    # aggregates total loss across observations
    total_loss = 0
    for n in range(N):
        z_n = Z[n]
        residual = y[n] - np.dot(z_n.T, beta)
        term = ((nu + 1) / 2) * np_autograd.log(
            1 + (np_autograd.exp(-2 * psi) / nu) * residual**2
        ) + psi
        total_loss += term

    # normalizes total loss by number of observations
    total_loss /= N

    # adds regularization term
    x = np_autograd.concatenate([np_autograd.array([psi]), beta])
    l2_term = (1 / (2 * N)) * np_autograd.sum(x**2)
    total_loss += l2_term

    return total_loss


# calculates gradient_vectors
def calculate_gradient_vectors(N, D, y, Z, nu):

    # initializes psi, beta, and x_0
    psi_0 = 0
    beta_0 = np.zeros(D)
    x_0 = np_autograd.concatenate([[psi_0], beta_0])

    # calculates gradient vector manually
    manual_gradient_vector = calculate_manual_gradient(x_0, y, Z, nu, N, N)

    # calculates gradient vector using autograd
    calculate_autograd_gradient = grad(loss_function)
    autograd_gradient_vector = calculate_autograd_gradient(x_0, y, Z, nu, N)

    print(f"Manual: {manual_gradient_vector}")
    print(f"Autograd: {autograd_gradient_vector}")


# calculates x_star
def calculate_x_star(N, D, y, Z, nu):

    res = minimize(loss_function, np.zeros(D + 1), args=(y, Z, nu, N))
    x_star = res.x

    print(f"x_star: {x_star}")

    return x_star


# runs SGD
def run_sgd(
    N, K, B, eta, eta_decay, alpha, x_init, x_star, D, y, Z, nu, rng, sgd_variant
):

    if sgd_variant == "base":
        # initializes parameter array
        params = np.zeros((K + 1, D + 1))
        params[0] = x_init

        errors = []

        # completes iterations
        for k in range(K):
            errors.append(np.sum((params[k] - x_star) ** 2))
            # samples batches
            inds = rng.choice(N, B, replace=False)
            y_sample = y[inds]
            Z_sample = Z[inds]
            params[k + 1] = params[k] - eta * calculate_manual_gradient(
                params[k], y_sample, Z_sample, nu, N, B
            )

        return params, None, errors

    elif sgd_variant == "iterate_avg":
        # initializes iterate average
        params = np.zeros((K + 1, D + 1))
        params[0] = x_init
        iterate_avg = x_init

        errors = []
        # completes iterations
        for k in range(K):
            window_size = (k + 1) // 2
            errors.append(np.sum((iterate_avg - x_star) ** 2))
            # samples batches
            inds = rng.choice(N, B, replace=False)
            y_sample = y[inds]
            Z_sample = Z[inds]

            # updates parameters
            # current_param = current_param - eta * calculate_manual_gradient(
            #     current_param, y_sample, Z_sample, v, N, B
            # )
            params[k + 1] = params[k] - eta * calculate_manual_gradient(
                params[k], y_sample, Z_sample, nu, N, B
            )

            # updates iterate average
            if k + 1 < 4:
                iterate_avg = params[k + 1]
            else:
                iterate_avg = np.mean(params[k + 1 - window_size : k + 2], axis=0)
            # iterate_avg = (k * iterate_avg + current_param) / (k + 1)

        return params, iterate_avg, errors

    elif sgd_variant == "poly_decay":
        # initializes current parameter
        current_param = x_init

        errors = []
        # completes iterations
        for k in range(K):
            errors.append(np.sum((current_param - x_star) ** 2))
            # samples batches
            inds = rng.choice(N, B, replace=False)
            y_sample = y[inds]
            Z_sample = Z[inds]

            # update parameters
            current_eta = eta_decay / ((k + 1) ** alpha)
            current_param = current_param - current_eta * calculate_manual_gradient(
                current_param, y_sample, Z_sample, nu, N, B
            )

        return current_param, current_eta, errors


# plots squared norm of errors vs epoch for each initial vector for the base and iterate average variants of SGD
def plot_sgd_base_iterate_avg_error(K, M, all_errors, sgd_variant, dataset):

    plt.figure(figsize=(6, 4))
    sorted_errors = dict(sorted(all_errors.items()))
    for a, errors in sorted_errors.items():
        plt.plot(range(len(errors)), errors, label=f"a={a}")

    step = 10 * K / M
    plt.xticks(np.arange(0, K + step, step), np.arange(0, M + 10, 10))
    plt.xlabel("Epoch")
    plt.ylabel(r"$\|x_k - x_*\|_2^2$")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid()

    if sgd_variant == "base":
        plt.title("SGD Error Comparsion")
        plt.tight_layout()
        directory = f"./images/{dataset}_dataset/base"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/rreg_error_comparison.png")
        plt.savefig("rreg_error_comparison.pdf")
    elif sgd_variant == "iterate_avg":
        plt.title("SGD-IA Error Comparison")
        plt.tight_layout()
        directory = f"./images/{dataset}_dataset/iterate_avg"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/rreg_error_comparison_iterate_avg.png")

    # plt.show()


# plots squared norm of errors vs epoch for each initial vector for the polynomial decay variant of SGD
def plot_sgd_polynomial_decay_error(K, M, alpha, alpha_errors, dataset):

    plt.figure(figsize=(6, 4))
    sorted_errors = dict(sorted(alpha_errors.items()))
    for a, errors in sorted_errors.items():
        plt.plot(range(len(errors)), errors, label=f"a={a}")

    step = 10 * K / M
    plt.xticks(np.arange(0, K + step, step), np.arange(0, M + 10, 10))
    plt.xlabel("Epoch")
    plt.ylabel(r"$\|x_k - x_*\|_2^2$")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid()
    plt.title(f"SGD-PD (α = {alpha}) Error Comparison")
    plt.tight_layout()

    directory = f"./images/{dataset}_dataset/poly_decay"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/rreg_error_comparison_poly_decay_a{alpha}.png")

    # plt.show()


# finds iteration where squared norm of error < D/4
def find_convergence_iteration(
    N, K, B, eta, eta_decay, alpha, x_init, x_star, D, y, Z, nu, rng, sgd_variant
):

    if sgd_variant == "base":
        # initializes parameter array
        params = np.zeros((K + 1, D + 1))
        params[0] = x_init

        # completes iterations
        for k in range(K):
            # samples batches
            inds = rng.choice(N, B)
            y_sample = y[inds]
            Z_sample = Z[inds]
            params[k + 1] = params[k] - eta * calculate_manual_gradient(
                params[k], y_sample, Z_sample, nu, N, B
            )

            # checks if error has converged
            if np.sum((params[k + 1] - x_star) ** 2) < D / 4:
                return k + 1

    elif sgd_variant == "iterate_avg":
        # initializes iterate average
        params = np.zeros((K + 1, D + 1))
        params[0] = x_init
        iterate_avg = x_init

        # completes iterations
        for k in range(K):
            window_size = k // 2
            # samples batches
            inds = rng.choice(N, B)
            y_sample = y[inds]
            Z_sample = Z[inds]

            # updates parameters
            # current_param = current_param - eta * calculate_manual_gradient(
            #     current_param, y_sample, Z_sample, nu, N, B
            # )
            params[k + 1] = params[k] - eta * calculate_manual_gradient(
                params[k], y_sample, Z_sample, nu, N, B
            )

            # checks if error has converged
            if np.sum((iterate_avg - x_star) ** 2) < D / 4:
                return k + 1

            # updates iterate average
            if k + 1 < 4:
                iterate_avg = params[k + 1]
            else:
                iterate_avg = np.mean(params[k + 1 - window_size : k + 2], axis=0)
            # iterate_avg = (k * iterate_avg + current_param) / (k + 1)

    elif sgd_variant == "poly_decay":
        # initializes iterate average
        current_param = x_init
        iterate_avg = x_init

        # completes iterations
        for k in range(K):
            # samples batches
            inds = rng.choice(N, B)
            y_sample = y[inds]
            Z_sample = Z[inds]

            # update parameters
            current_eta = eta_decay / ((k + 1) ** alpha)
            current_param = current_param - current_eta * calculate_manual_gradient(
                current_param, y_sample, Z_sample, nu, N, B
            )

            # checks if error has converged
            if np.sum((current_param - x_star) ** 2) < D / 4:
                return k + 1

            # update running average
            iterate_avg = (k * iterate_avg + current_param) / (k + 1)

    return None


# plots squared initial distance vs epochs until convergence for each initial vector for the base and iterate average variants of SGD
def plot_sgd_base_iterate_avg_convergence(
    N,
    B,
    initial_values,
    initial_distances,
    convergence_iterations,
    sgd_variant,
    dataset,
):

    plt.figure(figsize=(6, 4))
    for a in initial_values:
        if convergence_iterations[a] == None:
            continue
        epochs = convergence_iterations[a] * B / N
        plt.scatter(initial_distances[a], epochs, label=f"a={a}", alpha=0.6)

    plt.xlabel(r"$\|x_0 - x_*\|_2^2$")
    plt.xscale("log")
    plt.ylabel(r"Epochs until $\|x_k - x_*\|_2^2$ < D/4")
    plt.yscale("log")
    plt.grid()

    if sgd_variant == "base":
        plt.title("SGD Convergence Comparsion")
        plt.tight_layout()
        directory = f"./images/{dataset}_dataset/base"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/rreg_convergence_comparison.png")
        plt.savefig("rreg_convergence_comparison.pdf")
    elif sgd_variant == "iterate_avg":
        plt.title("SGD-IA Convergence Comparison")
        plt.tight_layout()
        directory = f"./images/{dataset}_dataset/iterate_avg"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/rreg_convergence_comparison_iterate_avg.png")

    # plt.show()


# plots squared initial distance vs epochs until convergence for each initial vector for the polynomial decay variants of SGD
def plot_sgd_poly_decay_convergence(
    N, B, alphas, initial_values, convergences, dataset
):

    plt.figure(figsize=(6, 4))
    for alpha in alphas:
        distances = []
        epochs = []
        for a in initial_values:
            if convergences[alpha]["convergence_iterations"][a] is None:
                continue
            distances.append(convergences[alpha]["initial_distances"][a])
            epochs.append(convergences[alpha]["convergence_iterations"][a] * B / N)

        plt.scatter(distances, epochs, alpha=0.6, label=f"α = {alpha}", marker="o")

        if len(distances) > 1:
            log_distances = np.log(distances)
            log_epochs = np.log(epochs)
            coeffs = np.polyfit(log_distances, log_epochs, 1)
            line = np.poly1d(coeffs)
            x_fit = np.linspace(min(log_distances), max(log_distances), 100)
            y_fit = line(x_fit)
            plt.plot(np.exp(x_fit), np.exp(y_fit))

    plt.xlabel(r"$\|x_0 - x_*\|_2^2$")
    plt.xscale("log")
    plt.ylabel(r"Epochs until $\|x_k - x_*\|_2^2$ < D/4")
    plt.yscale("log")
    plt.title("SGD-PD Convergence Comparison")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    directory = f"./images/{dataset}_dataset/poly_decay"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/rreg_convergence_comparison_poly_decay.png")

    # plt.show()


# wrapper for SGD for initial state x_0 = <a,...,a>
def process_sgd(
    N, K, B, eta, eta_decay, alpha, a, x_star, D, y, Z, nu, rng, sgd_variant
):

    start = time.perf_counter()
    x_init = np.array([a] * (D + 1))
    _, _, errors = run_sgd(
        N, K, B, eta, eta_decay, alpha, x_init, x_star, D, y, Z, nu, rng, sgd_variant
    )
    end = time.perf_counter()
    if sgd_variant == "base":
        print(
            f"Ran SGD (eta = {eta}) on x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )
    elif sgd_variant == "iterate_avg":
        print(
            f"Ran SGD w/ iterate average (eta = {eta}) on x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )
    elif sgd_variant == "poly_decay":
        print(
            f"Ran SGD w/ polynomial decay (eta = {eta_decay}, alpha = {alpha}) on x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )

    return a, errors


# wrapper for SGD w/ polynomial decay for initial state x_0 = <a,...,a>
def process_sgd_poly_decay(N, K, B, eta, eta_decay, alpha, a, x_star, D, y, Z, nu, rng):

    x_init = np.array([a] * (D + 1))
    start = time.perf_counter()
    x_init = np.array([a] * (D + 1))
    _, _, errors = run_sgd(
        N, K, B, eta, eta_decay, alpha, x_init, x_star, D, y, Z, nu, rng, "poly_decay"
    )
    end = time.perf_counter()
    print(
        f"Ran SGD w/ polynomial decay (eta = {eta_decay}, alpha = {alpha}) on x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
    )

    return (a, alpha), errors


# wrapper for finding SGD convergence iterations
def process_sgd_convergence(
    N, K, B, eta, eta_decay, alpha, a, x_star, D, y, Z, nu, rng, sgd_variant
):

    start = time.perf_counter()
    x_init = np.array([a] * (D + 1))
    d = np.sum((x_init - x_star) ** 2)
    k_star = find_convergence_iteration(
        N, K, B, eta, eta_decay, alpha, x_init, x_star, D, y, Z, nu, rng, sgd_variant
    )
    end = time.perf_counter()
    if sgd_variant == "base":
        print(
            f"Found SGD convergence iteration of x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )
    elif sgd_variant == "iterate_avg":
        print(
            f"Found SGD w/ iterate average convergence iteration of x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )
    elif sgd_variant == "poly_decay":
        print(
            f"Found SGD w/ polynomial decay (alpha = {alpha}) convergence iteration of x_0 = <{a},...,{a}> in {np.round(end - start, 2)} seconds"
        )

    return a, d, k_star


# processes the error and convergence rate across SGD variants
def process_sgd_variants(
    N,
    M,
    K,
    B,
    eta,
    eta_decay,
    alphas,
    x_star,
    D,
    y,
    Z,
    nu,
    rng,
    initial_values,
    sgd_variants,
    dataset,
):

    for sgd_variant in sgd_variants:

        if not sgd_variants[sgd_variant]:
            continue

        if sgd_variant == "base" or sgd_variant == "iterate_avg":
            all_errors = {}
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        process_sgd,
                        N,
                        K,
                        B,
                        eta,
                        eta_decay,
                        None,
                        a,
                        x_star,
                        D,
                        y,
                        Z,
                        nu,
                        rng,
                        sgd_variant,
                    ): a
                    for a in initial_values
                }
                for future in as_completed(futures):
                    a, errors = future.result()
                    all_errors[a] = errors
            # plots SGD error values
            plot_sgd_base_iterate_avg_error(K, M, all_errors, sgd_variant, dataset)
        elif sgd_variant == "poly_decay":
            for alpha in alphas:
                alpha_errors = {}
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            process_sgd,
                            N,
                            K,
                            B,
                            eta,
                            eta_decay,
                            alpha,
                            a,
                            x_star,
                            D,
                            y,
                            Z,
                            nu,
                            rng,
                            sgd_variant,
                        ): a
                        for a in initial_values
                    }
                    for future in as_completed(futures):
                        a, errors = future.result()
                        alpha_errors[a] = errors
                # plots SGD error values
                plot_sgd_polynomial_decay_error(K, M, alpha, alpha_errors, dataset)

        # collects SGD convergence iterations
        if sgd_variant == "base" or sgd_variant == "iterate_avg":
            initial_distances = {}
            convergence_iterations = {}
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        process_sgd_convergence,
                        N,
                        K,
                        B,
                        eta,
                        eta_decay,
                        None,
                        a,
                        x_star,
                        D,
                        y,
                        Z,
                        nu,
                        rng,
                        sgd_variant,
                    ): a
                    for a in initial_values
                }
                for future in as_completed(futures):
                    a, d, k_star = future.result()
                    initial_distances[a] = d
                    convergence_iterations[a] = k_star
            # plots SGD epochs until convergence
            plot_sgd_base_iterate_avg_convergence(
                N,
                B,
                initial_values,
                initial_distances,
                convergence_iterations,
                sgd_variant,
                dataset,
            )
        elif sgd_variant == "poly_decay":
            convergences = {}
            for alpha in alphas:
                initial_distances = {}
                convergence_iterations = {}
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(
                            process_sgd_convergence,
                            N,
                            K,
                            B,
                            eta,
                            eta_decay,
                            alpha,
                            a,
                            x_star,
                            D,
                            y,
                            Z,
                            nu,
                            rng,
                            sgd_variant,
                        ): a
                        for a in initial_values
                    }
                    for future in as_completed(futures):
                        a, d, k_star = future.result()
                        initial_distances[a] = d
                        convergence_iterations[a] = k_star
                convergences[alpha] = {
                    "initial_distances": initial_distances,
                    "convergence_iterations": convergence_iterations,
                }
            # plots SGD epochs until convergence
            plot_sgd_poly_decay_convergence(
                N, B, alphas, initial_values, convergences, dataset
            )


if __name__ == "__main__":

    # NOTE uncomment the dataset to be used
    # dataset given by professor
    # dataset = "given"
    # custom gpu dataset
    dataset = "gpu"

    # initializes common parameters
    print("\nInitializing common parameters")
    N, M, B, K, eta, eta_decay, alphas, rng, D, y, Z, nu, initial_values = (
        initialize_parameters(dataset)
    )

    # calculates gradient vectors
    # print("\nCalculating gradient vectors")
    # calculate_gradient_vectors(N, D, y, Z, nu)

    if dataset == "given":
        # print("\nCalculating x_star")
        # x_star = calculate_x_star(N, D, y, Z, nu)
        print("\nUsing previously calculated x_star")
        x_star = np.array(
            [
                -0.08119993,
                3.0456658,
                1.54969316,
                0.945425,
                0.86392264,
                0.52246674,
                0.59132863,
                0.33951322,
                0.48213225,
                0.20385502,
                0.44997904,
                0.14931394,
                0.33380023,
                0.2003957,
                0.22497222,
                0.15383423,
                0.30004565,
                0.03027903,
                0.31504017,
                0.05113866,
                0.19097387,
            ]
        )
        print(f"x_star: {x_star}")
    elif dataset == "gpu":
        # print("\nCalculating x_star")
        # x_star = calculate_x_star(N, D, y, Z, nu)
        print("\nUsing previously calculated x_star")
        x_star = np.array(
            [
                4.75342245,
                1.91284912,
                1.63898312,
                1.6123239,
                -7.27711527,
                -6.92710954,
                0.60217493,
                0.69364977,
                -0.75702951,
                2.02930482,
                0.72384499,
                -1.3184422,
                2.17312288,
                -11.72878692,
                -6.75589274,
            ]
        )
        print(f"x_star: {x_star}")

    # variants set to "True" will be compared
    sgd_variants = {"base": True, "iterate_avg": True, "poly_decay": True}

    # procecsses SGD variantsS
    print("\nProcessing SGD variants")
    process_sgd_variants(
        N,
        M,
        K,
        B,
        eta,
        eta_decay,
        alphas,
        x_star,
        D,
        y,
        Z,
        nu,
        rng,
        initial_values,
        sgd_variants,
        dataset,
    )
