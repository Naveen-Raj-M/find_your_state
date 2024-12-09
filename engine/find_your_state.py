import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, lognorm
from scipy.optimize import minimize
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from engine.args import Config

def weather(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the probability of temperature being between particular values for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of probabilities for each state.
    """
    # Directory containing the CSV files
    directory = 'data/weather/'

    # Initialize an empty dictionary to store the probability values
    probabilities = {}

    # Get a sorted list of all CSV files in the directory
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    # Iterate over the sorted list of files
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        
        # Read the CSV file and skip the first 4 lines
        df = pd.read_csv(filepath, skiprows=4)

        # Convert the first column to datetime and format it as short date
        df['Formatted Date'] = pd.to_datetime(df.iloc[:, 0]).dt.strftime('%Y-%m-%d')

        # Extract the temperature values
        temperature = df['Value'].to_numpy()

        # Plot the probability density function
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(temperature, bins=30, density=True, alpha=0.6, color='b')

        # Fit a normal distribution to the data
        mu, std = norm.fit(temperature)

        # Plot the normal distribution curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)

        # Add labels and title
        plt.xlabel('Temperature')
        plt.ylabel('Probability Density')
        plt.title(f'Probability Density Function of Temperature for {filename}')

        # Show mean and standard deviation in the plot
        plt.text(xmin + (xmax - xmin) * 0.05, max(p) * 0.9, f'Mean: {mu:.2f}\nStd Dev: {std:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Save the plot
        plot_filename = f'{filename[:-4]}.png'
        plt.savefig(os.path.join('results/weather', plot_filename))
        plt.close()

        # Calculate the probability of temperature being between the particular values
        probability = norm.cdf(cfg.weather.temperature, mu, std)
        probabilities[filename] = probability

    # Convert the dictionary of probabilities to a NumPy array
    probabilities_array = np.array(list(probabilities.values()))

    # Sort the probabilities dictionary by values in descending order
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    # Plot the probabilities for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_probabilities.keys(), sorted_probabilities.values())
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title(f'Probability of Temperature Being {cfg.weather.temperature} Degrees by State')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'probability_of_temperature.png'
    plt.savefig(os.path.join('results/weather', plot_filename))
    plt.close()

    return probabilities_array ** cfg.r.weather

def snowfall(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the probability of snowfall between specified limits for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of probabilities for each state.
    """
    # Path to the CSV file
    csv_file_path = 'data/snowfall/snowfall.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the 'days' column and calculate the probability of snowfall on any given day
    days = df['Days'].to_numpy()
    probabilities = days / 365

    # Calculate the probability of snowfall between lower_limit and upper_limit days in 365 days
    probability_between_limits = binom.cdf(cfg.snowfall.upper_bound, 365, probabilities) - binom.cdf(cfg.snowfall.lower_bound - 1, 365, probabilities)

    # Sort the probabilities and states by probability in descending order
    sorted_indices = np.argsort(probability_between_limits)[::-1]
    sorted_states = df['State'].to_numpy()[sorted_indices]
    sorted_probabilities = probability_between_limits[sorted_indices]

    # Plot the sorted probability between limits for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_states, sorted_probabilities)
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title(f'Probability of Snowfall Between {cfg.snowfall.lower_bound} days and {cfg.snowfall.upper_bound} days')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'probability_of_snowfall.png'
    plt.savefig(os.path.join('results/snowfall', plot_filename))
    plt.close()

    return probability_between_limits ** cfg.r.snowfall

def employment(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the normalized employment rate for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of normalized employment rates for each state.
    """
    # Path to the CSV file
    csv_file_path = 'data/employment/employment.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the 'employment_rate' column and convert it to a NumPy array
    employment_rate = df['employment_rate'].to_numpy()

    # Normalize the 'employment_rate' values to be within 0 and 1
    min_val = np.min(employment_rate)
    max_val = np.max(employment_rate)
    normalized_employment_rate = employment_rate / max_val

    # Sort the normalized employment rate and states by employment rate in descending order
    sorted_indices = np.argsort(normalized_employment_rate)[::-1]
    sorted_states = df['state'].to_numpy()[sorted_indices]
    sorted_employment_rate = normalized_employment_rate[sorted_indices]

    # Plot the sorted normalized employment rate for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_states, sorted_employment_rate)
    plt.xlabel('State')
    plt.ylabel('Normalized Employment Rate')
    plt.title('Normalized Employment Rate by State')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'normalized_employment_rate.png'
    plt.savefig(os.path.join('results/employment', plot_filename))
    plt.close()

    return normalized_employment_rate ** cfg.r.employment

def crime_rate(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the normalized crime rate for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of normalized crime rates for each state.
    """
    # Path to the CSV file
    csv_file_path = 'data/crime_rate/crime_rate.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the columns '2018' to '2022' and convert them to a NumPy array
    crime_rate_columns = df.loc[:, '2018':'2022'].to_numpy()

    # Find the average crime rate across the years
    average_crime_rate = np.mean(crime_rate_columns, axis=1)

    # Normalize the average crime rate values to be within 0 and 1
    min_val = np.min(average_crime_rate)
    max_val = np.max(average_crime_rate)
    normalized_crime_rate = 1 - (average_crime_rate / max_val)

    # Sort the normalized crime rate and states by crime rate in descending order
    sorted_indices = np.argsort(normalized_crime_rate)[::-1]
    sorted_states = df['state'].to_numpy()[sorted_indices]
    sorted_crime_rate = normalized_crime_rate[sorted_indices]

    # Plot the sorted normalized crime rate for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_states, sorted_crime_rate)
    plt.xlabel('State')
    plt.ylabel('Normalized Crime Rate')
    plt.title('Normalized Crime Rate by State')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'normalized_crime_rate.png'
    plt.savefig(os.path.join('results/crime_rate', plot_filename))
    plt.close()

    return normalized_crime_rate ** cfg.r.crime_rate

def cost_of_living(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the normalized cost of living index for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of normalized cost of living indices for each state.
    """
    # Path to the CSV file
    csv_file_path = 'data/cost_of_living/cost_of_living.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the 'Index' column and convert it to a NumPy array
    index = df['Index'].to_numpy()

    # Normalize the 'Index' values to be within 0 and 1
    min_val = np.min(index)
    max_val = np.max(index)
    normalized_index = 1 - (index / max_val)

    # Sort the normalized cost of living index and states by index in descending order
    sorted_indices = np.argsort(normalized_index)[::-1]
    sorted_states = df['State'].to_numpy()[sorted_indices]
    sorted_normalized_index = normalized_index[sorted_indices]

    # Plot the sorted normalized cost of living index for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_states, sorted_normalized_index)
    plt.xlabel('State')
    plt.ylabel('Normalized Cost of Living Index')
    plt.title('Normalized Cost of Living Index by State')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'normalized_cost_of_living_index.png'
    plt.savefig(os.path.join('results/cost_of_living', plot_filename))
    plt.close()

    return normalized_index ** cfg.r.cost_of_living
    
def state() -> list:
    """
    Extract the list of state names from the snowfall CSV file.

    Returns:
        list: List of state names.
    """
    # Path to the CSV file
    csv_file_path = 'data/snowfall/snowfall.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the 'State' column and convert it to a list
    state_names = df['State'].tolist()

    return state_names

def lognormal_params_from_percentiles(p20, p40, p50, p60, p80):
    """
    Estimate lognormal distribution parameters (mu, sigma) from percentiles.
    """
    z20, z40, z50, z60, z80 = map(norm.ppf, [0.20, 0.40, 0.50, 0.60, 0.80])

    def objective(params):
        mu, sigma = params
        return np.sum([
            (np.exp(mu + sigma * z20) - p20) ** 2,
            (np.exp(mu + sigma * z40) - p40) ** 2,
            (np.exp(mu + sigma * z50) - p50) ** 2,
            (np.exp(mu + sigma * z60) - p60) ** 2,
            (np.exp(mu + sigma * z80) - p80) ** 2,
        ])

    initial_guess = [np.log(p50), 0.5]
    result = minimize(objective, initial_guess, bounds=[(None, None), (1e-5, None)])
    mu, sigma = result.x
    return mu, sigma

def income(cfg: DictConfig) -> np.ndarray:
    """
    Calculate the probability of income being greater than a specified minimum for different states.

    Args:
        cfg (DictConfig): Configuration object containing parameters.

    Returns:
        np.ndarray: Array of probabilities for each state.
    """
    # Path to the CSV file
    csv_file_path = 'data/income/income.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the relevant columns for percentiles
    percentiles = ['p20', 'p40', 'p50', 'p60', 'p80']
    states = df['State'].tolist()

    # Initialize dictionaries to store mean, std dev, and probabilities
    mean_std_dict = {}
    probabilities = {}

    # Iterate over each state
    for state in states:
        # Extract the percentile values for the current state
        state_data = df[df['State'] == state][percentiles].values.flatten()

        # Extract the percentiles for the current state
        row = df[df['State'] == state].iloc[0]
        p20, p40, p50, p60, p80 = row[percentiles]

        # Estimate mu and sigma for the lognormal distribution
        mean, std_dev = lognormal_params_from_percentiles(p20, p40, p50, p60, p80)
        shape = std_dev
        loc = 0
        scale = np.exp(mean)

        # Store the mean and std dev in the dictionary
        mean_std_dict[state] = (mean, std_dev)

        # Plot the lognormal distribution
        plt.figure(figsize=(10, 6))
        x = np.linspace(min(state_data), max(state_data), 100)
        pdf = lognorm.pdf(x, shape, loc, scale)
        plt.plot(x, pdf, 'k', linewidth=2)

        # Plot vertical lines at 20, 40, 50, 60, and 80 percentiles
        percentiles_values = np.percentile(state_data, [20, 40, 50, 60, 80])
        percentiles_labels = ['20th percentile', '40th percentile', '50th percentile', '60th percentile', '80th percentile']
        for percentile, label in zip(percentiles_values, percentiles_labels):
            plt.axvline(x=percentile, color='r', linestyle='--', ymax=max(pdf)/max(pdf))
            plt.text(percentile, max(pdf) * 0.8, label, rotation=90, verticalalignment='center', color='r')

        # Add labels and title
        plt.xlabel('Income')
        plt.ylabel('Probability Density')
        plt.title(f'Lognormal Distribution of Income for {state}')

        # Show mean and standard deviation in the plot
        plt.text(min(state_data) + (max(state_data) - min(state_data)) * 0.05, max(pdf) * 0.9, f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Save the plot
        plot_filename = f'income_{state}.png'
        plt.savefig(os.path.join('results/income', plot_filename))
        plt.close()

        # Calculate the probability of income being greater than a particular value
        probability = 1 - lognorm.cdf(cfg.income.minimum, shape, loc, scale)
        probabilities[state] = probability

    # Convert the dictionary of probabilities to a NumPy array
    probabilities_array = np.array(list(probabilities.values()))

    # Sort the probabilities dictionary by values in descending order
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    # Plot the probabilities for different states
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_probabilities.keys(), sorted_probabilities.values())
    plt.xlabel('State')
    plt.ylabel('Probability')
    plt.title(f'Probability of Income Being Greater Than ${cfg.income.minimum} by State')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plot_filename = 'probability_of_income.png'
    plt.savefig(os.path.join('results/income', plot_filename))
    plt.close()

    return probabilities_array ** cfg.r.income


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config):
    weather_utility = weather(cfg)
    snowfall_utility = snowfall(cfg)
    employment_utility = employment(cfg)
    crime_rate_utility = crime_rate(cfg)
    cost_of_living_utility = cost_of_living(cfg)
    income_utility = income(cfg)

    # Calculate the weighted sum of utilities
    weighted_sum = ((cfg.weights.weather * weather_utility) + 
                    (cfg.weights.snowfall * snowfall_utility) + 
                    (cfg.weights.employment * employment_utility) + 
                    (cfg.weights.crime_rate * crime_rate_utility) + 
                    (cfg.weights.cost_of_living * cost_of_living_utility)
                    + (cfg.weights.income * income_utility))
    
    sum_of_weights = cfg.weights.weather + cfg.weights.snowfall + cfg.weights.employment + cfg.weights.crime_rate + cfg.weights.cost_of_living + cfg.weights.income
    weighted_sum_normalized = weighted_sum / sum_of_weights

    # Find the indices of the top three maximum utility values
    top_indices = np.argsort(weighted_sum_normalized)[-3:][::-1]

    state_names = state()

    # Print the top three maximum utility values and the corresponding state names
    for i, index in enumerate(top_indices):
        print(f'Top {i+1} Utility: {weighted_sum_normalized[index]}')
        print(f'State with Top {i+1} Utility: {state_names[index]}')
    # Sort the normalized weighted sum and state names by utility in descending order
    sorted_indices = np.argsort(weighted_sum_normalized)[::-1]
    sorted_state_names = np.array(state_names)[sorted_indices]
    sorted_weighted_sum_normalized = weighted_sum_normalized[sorted_indices]

    # Plot the normalized weighted sum of all the states
    plt.figure(figsize=(12, 8))
    plt.bar(sorted_state_names, sorted_weighted_sum_normalized)
    plt.xlabel('States')
    plt.ylabel('Resultant Utility')
    plt.title('Final Resultant Utility for All States')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot in the results directory
    plt.savefig('results/resultant_utility.png')
    plt.close()

if __name__ == "__main__":
    main()