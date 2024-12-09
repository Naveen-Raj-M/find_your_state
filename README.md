# Find Your State

## Configuration file

```yaml
defaults:
    - _self_
    - override hydra/hydra_logging: disabled  
    - override hydra/job_logging: disabled  

hydra:
    output_subdir: null  
    run:
        dir: .

# weather configuration
weather:
    temperature: 50
```
- This section configures the weather criteria. The `temperature` key sets the preferred average annual temperature.

```yaml
# snow configuration
snowfall:
    upper_bound: 30
    lower_bound: 5
```
- This section configures the snowfall criteria. The `upper_bound` and `lower_bound` keys set the acceptable range for the number of snowfall days.

```yaml
# income configuration
income:
    minimum: 80000
```
- This section configures the income criteria. The `minimum` key sets the minimum acceptable annual income in USD.

```yaml
# risk_affinity configuration
r:
    weather: 1
    snowfall: 0.5
    employment: 2
    crime_rate: 2
    cost_of_living: 1
    income: 3
```
- This section configures the risk affinity for each criterion. 
    `r` < 1 for risk-aversive utility
    `r` = 1 for risk-neutral utility
    `r` > 1 for risk-seeking utility

```yaml
# weights configuration
weights:
    weather: 10
    snowfall: 10
    employment: 20
    crime_rate: 20
    cost_of_living: 15
    income: 25
```
- This section configures the weights for each criterion. The values represent the relative importance of each criterion in the final decision-making process.
