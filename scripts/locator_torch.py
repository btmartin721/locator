import argparse
import copy
import os
import sys
import time
import pickle

import allel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from scipy import spatial
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--vcf", help="VCF with SNPs for all samples.")
parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
parser.add_argument(
    "--n_iter",
    type=int,
    required=True,
    help="Number of RandomizedGridSearch Iterations to use.",
)
parser.add_argument(
    "--matrix",
    help="tab-delimited matrix of minor allele counts with first column named 'sampleID'.\
        E.g., \
        \
        sampleID\tsite1\tsite2\t...\n \
        msp1\t0\t1\t...\n \
        msp2\t2\t0\t...\n ",
)
parser.add_argument(
    "--do_gridsearch",
    default=False,
    action="store_true",
    help="Do gridsearch to find best parameters? Defaults to False.",
)
parser.add_argument(
    "--sample_data",
    help="tab-delimited text file with columns\
        'sampleID \t x \t y'.\
        SampleIDs must exactly match those in the \
        VCF. X and Y values for \
        samples without known locations should \
        be NA.",
)

parser.add_argument(
    "--popmap",
    default=None,
    help="tab-delimited text file with columns sampleID \t populationID. There should not be a header line.",
)
parser.add_argument(
    "--train_split",
    default=0.9,
    type=float,
    help="0-1, proportion of samples to use for training. \
        default: 0.9 ",
)
parser.add_argument(
    "--val_split",
    default=0.2,
    type=float,
    help="0-1, proportion of samples to use for validation. \
        default: 0.2 ",
)
parser.add_argument(
    "--bootstrap",
    default=False,
    action="store_true",
    help="Run bootstrap replicates by retraining on bootstrapped data.",
)
parser.add_argument(
    "--nboots",
    default=50,
    type=int,
    help="number of bootstrap replicates to run.\
                    default: 50",
)
parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
parser.add_argument("--max_epochs", default=5000, type=int, help="default: 5000")
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="Learning rate to use. Defaults to 0.001",
)
parser.add_argument(
    "--l2_reg",
    default=0.0,
    type=float,
    help="L2 weight regularization to use. Defaults to 0.0 (no regularization)",
)
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last \
        improvement in validation loss. \
        default: 100",
)
parser.add_argument(
    "--min_mac",
    default=2,
    type=int,
    help="minimum minor allele count.\
        default: 2.",
)
parser.add_argument(
    "--max_SNPs",
    default=None,
    type=int,
    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.",
)
parser.add_argument(
    "--impute_missing",
    default=False,
    action="store_true",
    help="default: True (if False, all alleles at missing sites are ancestral)",
)
parser.add_argument(
    "--dropout_prop",
    default=0.25,
    type=float,
    help="proportion of weights to zero at the dropout layer. \
        default: 0.25",
)
parser.add_argument(
    "--nlayers",
    default=10,
    type=int,
    help="number of layers in the network. \
                        default: 10",
)
parser.add_argument(
    "--width",
    default=256,
    type=int,
    help="number of units per layer in the network\
                    default:256",
)
parser.add_argument("--out", help="file name stem for output")
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed for train/test splits and SNP subsetting.",
)
parser.add_argument("--gpu_number", default=None, type=str)
parser.add_argument(
    "--plot_history",
    default=True,
    type=bool,
    help="plot training history? \
                    default: True",
)
parser.add_argument(
    "--gnuplot",
    default=False,
    action="store_true",
    help="print acii plot of training history to stdout? \
                    default: False",
)
parser.add_argument(
    "--keep_weights",
    default=False,
    action="store_true",
    help="keep model weights after training? \
                    default: False.",
)
parser.add_argument(
    "--load_params",
    default=None,
    type=str,
    help="Path to a _params.json file to load parameters from a previous run.\
        Parameters from the json file will supersede all parameters provided \
        via command line.",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
)
# Parse arguments
args = parser.parse_args()

# Set seed and GPU
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


class MLPRegressor(nn.Module):
    """NN Model."""

    def __init__(self, input_size, width=256, nlayers=10, dropout_prop=0.2):
        super(MLPRegressor, self).__init__()
        self.input_layer = nn.Linear(input_size, width, device=device)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_prop)

        # Creating blocks of layers with residual connections
        self.blocks = nn.ModuleList()
        for _ in range(nlayers // 2):  # Handles even numbers of layers
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(width, width, device=device),
                    nn.BatchNorm1d(width, device=device),
                    nn.ELU(),
                    nn.Dropout(dropout_prop),
                    nn.Linear(width, width, device=device),
                    nn.BatchNorm1d(width, device=device),
                )
            )

        # Adding an additional layer if nlayers is odd
        self.extra_layer = None
        if nlayers % 2 != 0:
            self.extra_layer = nn.Sequential(
                nn.Linear(width, width, device=device),
                nn.BatchNorm1d(width, device=device),
                nn.ELU(),
                nn.Dropout(dropout_prop),
            )

        self.output_layer = nn.Linear(width, 2, device=device)

    def forward(self, x):
        x = self.elu(self.input_layer(x))
        # Applying residual blocks
        for block in self.blocks:
            residual = x
            x = self.elu(block(x) + residual)  # Add the residual (skip connection)
        if self.extra_layer is not None:
            x = self.extra_layer(x)
        x = self.dropout(x)
        return self.output_layer(x)


class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss


def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        positions = callset["variants/POS"]
    elif args.vcf is not None:
        print("reading VCF")
        vcf = allel.read_vcf(args.vcf, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif args.matrix is not None:
        gmat = pd.read_csv(args.matrix, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")
        for i in range(
            gmat.shape[0]
        ):  # kludge to get haplotypes for reading in to allel.
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))
        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
    return genotypes, samples


def sort_samples(genotypes, samples):
    sample_data = pd.read_csv(args.sample_data, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    # sort loc table so samples are in same order as vcf samples
    sample_data = sample_data.reindex(np.array(samples))

    if args.popmap is not None:
        popmap_data = pd.read_csv(
            args.popmap, sep="\t", names=["sampleID", "populationID"]
        )
        popmap_data["sampleID2"] = popmap_data["sampleID"]
        popmap_data.set_index("sampleID", inplace=True)
        populations = popmap_data.astype(str)
        popmap_data = popmap_data.reindex(np.array(samples))

    # check that all sample names are present
    if not all([sample_data["sampleID2"][i] == x for i, x in enumerate(samples)]):
        raise ValueError("sample ordering failed! Check that sample IDs match the VCF.")

    # Check that all sample names are present.
    if args.popmap is not None:
        if not all([popmap_data["sampleID2"][i] == x for i, x in enumerate(samples)]):
            raise ValueError(
                "population ordering failed! Check that the popmap sample IDs match the VCF"
            )
    locs = np.array(sample_data[["x", "y"]])
    print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
    print(f"loaded {len(popmap_data['populationID'].unique())} unique populations\n\n")
    return (sample_data, locs, popmap_data)


def replace_md(genotypes):
    """replace missing sites with binomial(2,mean_allele_frequency)"""

    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes):
    """Filter and impute SNPs"""
    print("filtering SNPs")
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]
    if not args.min_mac == 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= args.min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]
    if args.impute_missing:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]
    if not args.max_SNPs == None:
        ac = ac[np.random.choice(range(ac.shape[0]), args.max_SNPs, replace=False), :]
    print("running on " + str(len(ac)) + " genotypes after filtering\n\n\n")
    return ac


def normalize_locs(locs):
    """Normlize locations, ignoring NaN"""
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, locs


def train_model(
    train_loader,
    val_loader,
    model,
    trial,
    criterion,
    optimizer,
    epochs,
    patience=100,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=args.patience // 6,
):
    """Train the pytorch model."""
    early_stopping = EarlyStopping(patience=patience)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=False,
    )

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Early Stopping and LR Scheduler
        early_stopping(avg_val_loss, model)
        lr_scheduler.step(avg_val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if trial is not None:
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return model, train_losses, val_losses


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function."""
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, axis=1)).mean()


def objective(
    trial,
    X_train,
    train_loader,
    test_loader,
    val_loader,
    device,
    epochs,
    patience,
    lr_scheduler_factor,
    lr_scheduler_patience,
):
    """Optuna hyperparameter tuning."""
    # Optuna hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    width = trial.suggest_int("width", 32, 512)
    nlayers = trial.suggest_int("nlayers", 2, 20)
    dropout_prop = trial.suggest_float("dropout_prop", 0.0, 0.5)
    l2_weight = trial.suggest_float("l2_weight", 1e-6, 1e-1, log=True)

    # Model, loss, and optimizer
    model = MLPRegressor(
        input_size=X_train.shape[1],
        width=width,
        nlayers=nlayers,
        dropout_prop=dropout_prop,
    ).to(device)
    criterion = euclidean_distance_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)

    # Train model
    trained_model, train_losses, val_losses = train_model(
        train_loader,
        test_loader,
        model,
        trial,
        criterion,
        optimizer,
        epochs,
        patience,
        lr_scheduler_factor,
        lr_scheduler_patience,
    )

    # Evaluate model
    trained_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def predict_locations(
    model,
    data_loader,
    sdlong,
    meanlong,
    sdlat,
    meanlat,
    device,
    verbose=True,
):
    """
    Predict locations using the trained model and evaluate predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model for predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for prediction.
        sdlong (float): Standard deviation of longitude in the training dataset.
        meanlong (float): Mean longitude in the training dataset.
        sdlat (float): Standard deviation of latitude in the training dataset.
        meanlat (float): Mean latitude in the training dataset.
        sample_data (pandas.DataFrame): Input sample_data object.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        verbose (bool): If True, prints out a message upon completion.

    Returns:
        pandas.DataFrame: DataFrame with predicted locations and corresponding sample IDs.
    """
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            ground_truth.append(target.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # Rescale predictions and ground truth to original scale
    rescaled_preds = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in predictions]
    )
    rescaled_truth = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in ground_truth]
    )

    # Evaluate predictions
    r2_long = np.corrcoef(rescaled_preds[:, 0], rescaled_truth[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(rescaled_preds[:, 1], rescaled_truth[:, 1])[0][1] ** 2
    mean_dist = np.mean(
        [
            spatial.distance.euclidean(rescaled_preds[x, :], rescaled_truth[x, :])
            for x in range(len(rescaled_preds))
        ]
    )
    median_dist = np.median(
        [
            spatial.distance.euclidean(rescaled_preds[x, :], rescaled_truth[x, :])
            for x in range(len(rescaled_preds))
        ]
    )

    if verbose:
        print(f"R2(x) = {r2_long}\nR2(y) = {r2_lat}")
        print(f"Mean Validation Error (Euclidean Distance) = {mean_dist}")
        print(f"Median Validation Error (Euclidean Distance) = {median_dist}")

    # return the evaluation metrics along with the predictions
    metrics = {
        "r2_long": r2_long,
        "r2_lat": r2_lat,
        "mean_dist": mean_dist,
        "median_dist": median_dist,
    }

    return rescaled_preds, metrics


def plot_bootstrap_aggregates(df, filename):
    """Make a KDE plot with bootstrap distributions."""
    plt.figure(figsize=(10, 6))

    df_r2 = df[["r2_long", "r2_lat"]]

    df_melt = df_r2.melt()

    # Histogram for Longitude
    sns.kdeplot(
        data=df_melt, x="value", hue="variable", fill=True, palette="Set2", legend=True
    )
    plt.title("Distribution of Bootstrapped Error", fontsize=20)
    plt.xlabel("Euclidean Distance Error", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.savefig(filename, facecolor="white", bbox_inches="tight")
    plt.close()


def plot_history(train_loss, val_loss, filename):
    """Plot training and validation loss."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def bootstrap_training_generator(
    train_loader,
    test_loader,
    nboots,
    criterion,
    epochs,
    device,
    weights,
    width,
    nlayers,
    dropout_prop,
    lr,
    l2_weight,
):
    """
    Generator for training models on bootstrapped samples.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        nboots (int): Number of bootstrap samples to create.
        criterion: Loss function used for training.
        epochs (int): Number of epochs for training each model.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        weights (np.ndarray): Class weights for imbalanced sampling.
        width (int): Number of neurons in hidden layers.
        nlayers (int): Number of hidden layers.
        dropout_prop (float): Dropout proportion to reduce overfitting.
        lr (float): Learning rate for optimizer.
        l2_weight (float): L2 regularization weight (weight decay).

    Yields:
        Trained model for each bootstrap sample.
    """
    for _ in range(nboots):
        # Resampling with replacement
        resampled_indices = torch.randint(
            0, len(train_loader.dataset), (len(train_loader.dataset),)
        )

        # Obtain the weights corresponding to the resampled indices
        resampled_weights = weights[resampled_indices]

        # Create a WeightedRandomSampler with the resampled weights
        sampler = torch.utils.data.WeightedRandomSampler(
            resampled_weights, len(resampled_weights), replacement=True
        )

        # Create a Subset of the dataset corresponding to the resampled indices
        resampled_dataset = Subset(train_loader.dataset, resampled_indices)

        # Create a DataLoader for the resampled dataset with the new sampler
        resampled_loader = DataLoader(
            resampled_dataset, batch_size=args.batch_size, sampler=sampler
        )

        # Reinitialize the model and optimizer each bootstrap
        model = MLPRegressor(
            input_size=train_loader.dataset.tensors[0].shape[1],
            width=width,
            nlayers=nlayers,
            dropout_prop=dropout_prop,
        ).to(device)

        criterion = euclidean_distance_loss
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)

        # Train the model.
        trained_model, train_losses, val_losses = train_model(
            resampled_loader,
            test_loader,
            model,
            None,
            criterion,
            optimizer,
            epochs,
            args.patience,
            0.5,
            args.patience // 6,
        )

        yield trained_model, train_losses, val_losses


def split_train_test(ac, locs, train_split, val_split, seed):
    """
    Split data into training, validation, and testing sets, handling NaN values in locations.

    Args:
        ac (numpy.ndarray): Genotype data.
        locs (numpy.ndarray): Location data.
        train_split (float): Proportion of the data to be used for training.
        val_split (float): Proportion of the training data to be used for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: tuple containing indices and data for training, validation, and testing.
    """
    # Identify indices with non-NaN locations
    train_val_indices = np.argwhere(~np.isnan(locs[:, 0])).flatten()
    pred_indices = np.array([x for x in range(len(locs)) if x not in train_val_indices])

    # Split non-NaN samples into training + validation and test sets
    train_val_indices, test_indices = train_test_split(
        train_val_indices, train_size=train_split, random_state=seed
    )

    # Split training data into actual training and validation sets
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_split, random_state=seed
    )

    # Prepare genotype and location data for training, validation, testing, and
    #  prediction
    traingen = np.transpose(ac[:, train_indices])
    valgen = np.transpose(ac[:, val_indices])
    trainlocs = locs[train_indices]
    vallocs = locs[val_indices]
    testgen = np.transpose(ac[:, test_indices])
    testlocs = locs[test_indices]
    predgen = np.transpose(ac[:, pred_indices])
    return (
        train_indices,
        test_indices,
        val_indices,
        pred_indices,
        traingen,
        testgen,
        trainlocs,
        testlocs,
        valgen,
        vallocs,
        predgen,
    )


def get_class_weights(popmap_data, train_indices):
    """Get class weights for torch.utils.data.WeightedRandomSampler."""
    weight_train = popmap_data["populationID"].iloc[train_indices].to_numpy()

    unique_pops = np.unique(weight_train)
    pop_to_index = {pop: idx for idx, pop in enumerate(unique_pops)}

    class_sample_count = np.array(
        [len(np.where(weight_train == pop)[0]) for pop in unique_pops]
    )

    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[pop_to_index[pop]] for pop in weight_train])

    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight


def write_outputs(study):
    """Write Optuna study to file."""
    df = study.trials_dataframe()
    df.to_csv(f"{args.out}_trials_df.csv", header=True)

    with open(f"{args.out}_sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)

    with open(f"{args.out}_best_score.txt", "w") as fout:
        fout.write(str(study.best_value))

    with open(f"{args.out}_best_params.pkl", "wb") as fout:
        pickle.dump(study.best_params, fout)

    with open(f"{args.out}_best_trials.pkl", "wb") as fout:
        pickle.dump(study.best_trials, fout)

    with open(f"{args.out}_best_overall_trial.pkl", "wb") as fout:
        pickle.dump(study.best_trial, fout)

    with open(f"{args.out}_all_trials.pkl", "wb") as fout:
        pickle.dump(study.trials, fout)


def make_plots(study):
    """Visualize Optuna search."""
    if not optuna.visualization.is_available():
        return

    importance_fig = optuna.visualization.plot_param_importances(study)
    importance_fig.write_image(f"{args.out}_param_importances.png")

    edf_fig = optuna.visualization.plot_edf(study, target_name="Location Error")
    edf_fig.write_image(f"{args.out}_edf.png")

    par_fig = optuna.visualization.plot_parallel_coordinate(
        study, target_name="Location Error"
    )
    par_fig.write_image(f"{args.out}_parallel_coordinates.png")

    slice_fig = optuna.visualization.plot_slice(study, target_name="Location Error")
    slice_fig.write_image(f"{args.out}_slices.png")

    tl_fig = optuna.visualization.plot_timeline(study)
    tl_fig.write_image(f"{args.out}_timeline.png")

    rank_fig = optuna.visualization.plot_rank(study, target_name="Location Error")
    rank_fig.write_image(f"{args.out}_rank.png")

    ctr_fig = optuna.visualization.plot_contour(study, target_name="Location Error")
    ctr_fig.write_image(f"{args.out}_contour.png")

    hist_fig = optuna.visualization.plot_optimization_history(
        study, target_name="Location Error"
    )
    hist_fig.write_image(f"{args.out}_opt_history.png")


def main():
    """Run the script."""
    # Data Preparation
    genotypes, samples = load_genotypes()
    sample_data, locs, popmap_data = sort_samples(genotypes, samples)
    meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
    ac = filter_snps(genotypes)

    (
        train_indices,
        test_indices,
        val_indices,
        pred_indices,
        X_train,
        X_test,
        y_train,
        y_test,
        X_val,
        y_val,
        X_pred,
    ) = split_train_test(ac, locs, args.train_split, args.val_split, args.seed)

    # Make torch datasets.
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.float),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    )

    # Do weighted sampling per population if popmap provided.
    if args.popmap is not None:
        samples_weight = get_class_weights(popmap_data, train_indices)
    else:
        samples_weight = None

    # For class imbalance (populations).
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weight, num_samples=len(samples_weight), replacement=True
    )

    # Create dataloaders.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=weighted_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Do a hyperparameter search.
    if args.do_gridsearch:
        # Optuna Optimization with TPE sampler and MedianPruner
        sampler = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{args.out}_optuna.db",
            load_if_exists=True,
            study_name=f"{args.out}_torch_study",
        )

        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                train_loader,
                test_loader,
                val_loader,
                device,
                args.max_epochs,
                args.patience,
                0.5,
                args.patience // 6,
            ),
            n_trials=args.n_iter,
            n_jobs=-1,
            show_progress_bar=False,
        )
        best_trial = study.best_trial

        make_plots(study)
        write_outputs(study)

        # Define the model architecture
        model_prototype = MLPRegressor(
            input_size=X_train.shape[1],
            width=best_trial.params["width"],
            nlayers=best_trial.params["nlayers"],
            dropout_prop=best_trial.params["dropout_prop"],
        ).to(device)
        criterion = euclidean_distance_loss
        optimizer_prototype = optim.Adam(
            model_prototype.parameters(),
            lr=best_trial.params["lr"],
            weight_decay=best_trial.params["l2_weight"],
        )

        width = best_trial.params["width"]
        nlayers = best_trial.params["nlayers"]
        dropout_prop = best_trial.params["dropout_prop"]
        lr = best_trial.params["lr"]
        l2_weight = best_trial.params["l2_weight"]

    else:
        model_prototype = MLPRegressor(
            input_size=X_train.shape[1],
            width=args.width,
            nlayers=args.nlayers,
            dropout_prop=args.dropout_prop,
        ).to(device)

        criterion = euclidean_distance_loss
        optimizer_prototype = optim.Adam(
            model_prototype.parameters(),
            lr=args.learning_rate,
            weight_decay=args.l2_reg,
        )
        best_trial = None
        width = args.width
        nlayers = args.nlayers
        dropout_prop = args.dropout_prop
        lr = args.learning_rate
        l2_weight = args.l2_reg

    # Bootstrapping or Standard Training
    if args.bootstrap:
        # Bootstrapping
        bootstrap_gen = bootstrap_training_generator(
            train_loader,
            test_loader,
            args.nboots,
            criterion,
            args.max_epochs,
            device,
            samples_weight,
            width,
            nlayers,
            dropout_prop,
            lr,
            l2_weight,
        )

        bootstrap_preds = []
        bootstrap_preds_dfs = []
        boot_metrics = []
        for boot, (trained_model, train_losses, val_losses) in enumerate(bootstrap_gen):
            print(f"Boostrap {boot + 1} of {args.nboots}")

            # Save or evaluate the trained model
            torch.save(trained_model.state_dict(), f"bootstrap_model_{boot}.pt")

            # Predict locations
            val_preds, val_metrics = predict_locations(
                trained_model,
                val_loader,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                device,
            )
            boot_metrics.append(val_metrics)
            bootstrap_preds.append(val_preds)

            with open(f"{args.out}_boot{boot}_metrics.txt", "w") as fout:
                for k, v in val_metrics.items():
                    fout.write(f"{k},{v}\n")

            bootstrap_preds_dfs.append(
                write_pred_locations(
                    val_preds,
                    val_indices,
                    sample_data,
                    f"{args.out}_boot{boot}_predlocs.txt",
                )
            )

        plot_bootstrap_aggregates(
            pd.DataFrame.from_dict(boot_metrics),
            f"{args.out}_bootstrap_distribution.png",
        )

    # Standard training
    # Add validation loader if using early stopping
    best_model, train_losses, val_losses = train_model(
        train_loader,
        test_loader,
        model_prototype,
        best_trial,
        criterion,
        optimizer_prototype,
        args.max_epochs,
        args.patience,
        0.5,
        args.patience // 6,
    )

    # Evaluate the best model
    val_preds, val_metrics = predict_locations(
        best_model, val_loader, sdlong, meanlong, sdlat, meanlat, device
    )

    with open(f"{args.out}_validation_metrics.txt", "w") as fout:
        for k, v in val_metrics.items():
            fout.write(f"{k},{v}\n")

    val_preds = write_pred_locations(
        val_preds, val_indices, sample_data, f"{args.out}_validation_predlocs.txt"
    )

    # Plot training history
    plot_history(train_losses, val_losses, f"{args.out}_training_history.png")

    # Convert X_pred to a PyTorch tensor and move it to the correct device
    pred_tensor = torch.tensor(X_pred, dtype=torch.float).to(device)

    with torch.no_grad():
        # Make predictions
        pred_locations_scaled = best_model(pred_tensor)

    # rescale the predictions back to the original range
    pred_locations = np.array(
        [
            [x[0] * sdlong + meanlong, x[1] * sdlat + meanlat]
            for x in pred_locations_scaled.cpu().numpy()
        ]
    )

    real_preds = write_pred_locations(
        pred_locations,
        pred_indices,
        sample_data,
        f"{args.out}_real_predlocs.txt",
    )


def write_pred_locations(pred_locations, pred_indices, sample_data, filename):
    """write predicted locations to file."""
    pred_locations_df = pd.DataFrame(pred_locations, columns=["x", "y"])
    sample_data = sample_data.reset_index()
    sample_data = sample_data.iloc[pred_indices].copy()
    pred_locations_df.reset_index(drop=True)
    pred_locations_df["sampleID"] = sample_data["sampleID"].tolist()
    pred_locations_df = pred_locations_df[["sampleID", "x", "y"]]
    pred_locations_df.to_csv(filename, header=True, index=False)
    return pred_locations_df


if __name__ == "__main__":
    main()
