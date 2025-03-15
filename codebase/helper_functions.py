'''
Fucntions to be applied for processing datasets for drought index development
'''

# Standardised Anomaly

# Function to calculate standardized anomalies
def calc_standardized_anomalies(data, var_name):
    '''To remove climatology, calculate the monthly climatology (long-term mean and standard deviation for each month) and then compute standardized anomalies for each variable (tws, precip, et) per subregion.'''
    # Group by month and calculate climatological mean and std
    monthly_clim = data[var_name].groupby('time.month').mean('time')
    monthly_std = data[var_name].groupby('time.month').std('time')
    
    # Compute anomalies: (value - mean) / std
    anomalies = data[var_name].groupby('time.month') - monthly_clim
    standardized_anomalies = anomalies.groupby('time.month') / monthly_std
    
    return standardized_anomalies


# Marginal Distribution fitting

# Function to compute ECDF with a common mask
def compute_ecdf_statsmodels_consistent(ds):
    '''Parameters:
        ds (xarray.Dataset): Input dataset containing 'tws_anom', 'precip_anom', and 'et_anom'.

    Returns:
        tws_cdf (xarray.DataArray): ECDF values for TWS anomaly.
        precip_cdf (xarray.DataArray): ECDF values for precipitation anomaly.
        et_cdf (xarray.DataArray): ECDF values for ET anomaly.
    '''
    # Convert the selected variables to a single DataArray with 'variable' dimension
    data_vars = ds[['tws_anom', 'precip_anom', 'et_anom']].to_array(dim='variable')
    
    # Stack spatial and temporal dimensions
    stacked_data = data_vars.stack(all_points=['time', 'y', 'x'])
    
    # Create a common mask where all variables are non-NaN
    common_mask = stacked_data.notnull().all(dim='variable')
    valid_data = stacked_data.where(common_mask, drop=True)
    # print(valid_data)
    
    # Compute ECDF for each variable
    ecdf_dict = {}
    for var_name in ['tws_anom', 'precip_anom', 'et_anom']:
        flat_data = valid_data.sel(variable=var_name).values
        ecdf = ECDF(flat_data)
        cdf_values = ecdf(flat_data)
        ecdf_dict[var_name] = xr.DataArray(
            cdf_values,
            coords={'all_points': valid_data.sel(variable=var_name).coords['all_points']},
            dims=['all_points']
        ).unstack()
    
    return ecdf_dict['tws_anom'], ecdf_dict['precip_anom'], ecdf_dict['et_anom']


# Function to test and select the best copula
def fit_best_copula(tws_cdf, precip_cdf, et_cdf):
    """
    Fit and select the best copula based on AIC.

    Parameters:
        tws_cdf (xarray.DataArray): CDF values for TWS anomaly.
        precip_cdf (xarray.DataArray): CDF values for precipitation anomaly.
        et_cdf (xarray.DataArray): CDF values for ET anomaly.

    Returns:
        best_copula: The best-fitting copula object.
        results (dict): Dictionary containing metrics for all tested copulas.
    """
    
    uniform_data = pd.DataFrame({
        'tws': tws_cdf.values,
        'precip': precip_cdf.values,
        'et': et_cdf.values
    }).dropna()
    
    copulas = {
        'Gaussian': GaussianCopula(dim=3),
        'Clayton': ClaytonCopula(dim=3),
        'Frank': FrankCopula(dim=3),
        'Gumbel': GumbelCopula(dim=3),
        # 'tCopula': StudentCopula(dim=3)
    }
    
    results = {}
    for name, copula in copulas.items():
        try:
            copula.fit(uniform_data)
            log_lik = copula.log_lik(uniform_data)
            if name in ['Clayton', 'Gumbel', 'Frank']:
                n_params = np.array(copula.params).size
            elif name == 'Gaussian':
                n_params = copula.params.size
            aic = -2 * log_lik + 2 * n_params
            bic = -2 * log_lik + np.log(uniform_data.shape[0]) * n_params
            results[name] = {
                "Log-Likelihood": log_lik,
                "AIC": aic,
                "BIC": bic,
                "Parameters": copula.params,
                "copula": copula
            }
            print(f"{name} Copula: Log-Likelihood={log_lik:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")
        except Exception as e:
            print(f"Error fitting {name} copula: {e}")
            results[name] = {
                "Log-Likelihood": -np.inf,
                "AIC": np.inf,
                "BIC": np.inf,
                "Parameters": None,
                "copula": None
            }
    
    best_copula_name = min(results, key=lambda x: results[x]['AIC'])
    best_copula = results[best_copula_name]['copula']
    print(f"Best copula (by AIC): {best_copula_name} (AIC: {results[best_copula_name]['AIC']:.2f})")
    return best_copula, results


# Lookup and apply functions
def lookup_best_copula(subregion_id):
    if subregion_id not in best_copulas:
        raise ValueError(f"Subregion {subregion_id} not found in processed data.")
    
    best_copula = best_copulas[subregion_id]
    metrics = copula_metrics[subregion_id]
    best_copula_name = min(metrics, key=lambda x: metrics[x]['AIC'])
    
    print(f"Subregion {subregion_id}:")
    print(f"Best Copula: {best_copula_name}")
    print(f"Log-Likelihood: {metrics[best_copula_name]['Log-Likelihood']:.2f}")
    print(f"AIC: {metrics[best_copula_name]['AIC']:.2f}")
    print(f"BIC: {metrics[best_copula_name]['BIC']:.2f}")
    print(f"Parameters: {metrics[best_copula_name]['Parameters']}")
    return best_copula