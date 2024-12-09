# find_your_state

## Description
The `find_your_state.py` script is designed to help users identify their state based on certain criteria defined in the configuration file. It processes the input data and matches it against the predefined rules to determine the state.

## Cloning the Repository and Setting Up the Virtual Environment
To get started with the project, follow these steps:

1. Clone the git repository:
    ```sh
    git clone https://github.com/Naveen-Raj-M/find_your_state.git
    ```

2. Navigate to the project directory:
    ```sh
    cd find_your_state
    ```

3. Build the virtual environment:
    ```sh
    source build_venv.sh
    ```

## Executing the Script
To execute the `find_your_state.py` script, use the following command:
```sh
python3 -m engine.find_your_state --config-path ../ --config-name config.yaml
```