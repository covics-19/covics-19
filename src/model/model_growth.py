import csv
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
import sys
import ast
import matplotlib.pyplot as plt

sys.path.insert(0, '../')
from utils import populate_results
from utils import fetch_hopkins

country_to_code_file = '../../data/external/country_to_code_dict.txt'
with open(country_to_code_file, 'r') as data:
    string = data.read()
    country_code_dict = ast.literal_eval(string)

# --------------------- DataFrame labels --------------------- #
COUNTRY_LABEL = 'Country'
PROVINCE_LABEL = 'Province'

def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))


def exponential(t, a, b, c):
    return a * np.exp(b * t) + c


def plot_base(x, y):
    plt.figure(figsize=(10,5))
    plt.plot(x, y, 'ko', label="Observed infections")


def plot_details(country, current_date):
    plt.title(country + ' Cumulative Confirmed COVID-19 Cases. (Updated on '+current_date+')', fontsize="x-large")
    plt.xlabel('Days', fontsize="x-large")
    plt.ylabel('Total Confirmed Cases', fontsize="x-large")
    plt.legend(fontsize="x-large")
    plt.ticklabel_format(style = 'plain')


def plot_scale_and_save(country):
    plt.yscale('linear')
    plt.savefig('../../data/figures/'+country+'_linear.png')
    # plt.yscale('log')
    # plt.savefig('../../data/figures/'+country+'_log.png')


def logistic_graph(x, y, lpopt, x_log, y_log, country, current_date):
    plot_base(x, y)
    plt.plot(x, logistic(x, *lpopt), 'g--', label="Expected infections (logistic)")
    plt.plot(x_log, y_log, 'y--', label="Predicted infections (logistic)") # plot predictions
    plot_details(country, current_date)
    plot_scale_and_save(country)


def exponential_graph(x, y, epopt, x_exp, y_exp, country, current_date):
    plot_base(x, y)
    plt.plot(x, exponential(x, *epopt), 'r--', label="Expected infections (exponential)")
    plt.plot(x_exp, y_exp, 'b--', label="Predicted infections (exponential)") # plot predictions
    plot_details(country, current_date)
    plot_scale_and_save(country)


def get_country_cases_time_series(confirmed_cases_df, country):
    # filter down to region rows from the country of interest
    country_cases = confirmed_cases_df[confirmed_cases_df['Country'] == country]
    # take only the columns of interest (cases by date)
    date_columns = country_cases.drop(['Province', 'Country'], axis = 1)
    # sum the different regions in the country to create the time series frame
    time_series = date_columns.T.sum(axis = 1)
    time_series_frame = pd.DataFrame(time_series)
    time_series_frame.columns = ['Cases']
    # drop the early dates for a country, before they had a case
    time_series_frame = time_series_frame.loc[time_series_frame['Cases'] > 0]
    return time_series_frame


def find_recent_doubling_time(current, lastweek):
    if current > lastweek:
        ratio = current/lastweek
        dailypercentchange = round( 100 * (pow(ratio, 1/7) - 1), 1)
        recentdbltime = round( 7 * np.log(2) / np.log(ratio), 1)
        return ratio, dailypercentchange, recentdbltime


def try_logistic(x, y, days, verbose=False):
    logisticworked = False
    lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
    lerror = np.sqrt(np.diag(lpcov))

    # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
    ldoubletime = np.log(2)/(lpopt[1]/2)
    # standard error
    ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1]/lpopt[1])
    # calculate R^2
    residuals = y - logistic(x, *lpopt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    logisticr2 = 1 - (ss_res / ss_tot)

    if logisticr2 > 0.95:
        logisticworked = True
        # make predictions
        day_now = y.size-1 # what day we are on now
        future_day = day_now + days # how many days in the future we are predicting
        preds_log = [logistic(t,lpopt[0],lpopt[1],lpopt[2],lpopt[3]) for t in list(range(day_now,future_day,1))] # do pred
        x_log = list(range(day_now, future_day, 1))
        y_log = preds_log

        if verbose:
            print('\n** Based on Logistic Fit**\n')
            print('\tR^2:', logisticr2)
            print('\tDoubling Time (during middle of growth): ', round(ldoubletime,2), '(±', round(ldoubletimeerror,2),') days')
            print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            print('\tPredicted number of infections (logistic growth):',round(preds_log[-1]))
    
    return logisticworked, lpopt, ldoubletime, ldoubletimeerror, logisticr2, preds_log, x_log, y_log


def try_exponential(x, y, days, verbose=False):
    exponentialworked = False
    epopt, epcov = curve_fit(exponential, x, y, bounds=([0,0,-100],[100,0.9,100]), maxfev=10000)
    eerror = np.sqrt(np.diag(epcov))

    # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
    edoubletime = np.log(2)/epopt[1]
    # standard error
    edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1]/epopt[1])

    # calculate R^2
    residuals = y - exponential(x, *epopt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    expr2 = 1 - (ss_res / ss_tot)

    if expr2 > 0.95:
        exponentialworked = True
        # make predictions
        day_now = y.size-1 # what day we are on now
        future_day = day_now + days # how many days in the future we are predicting

        preds_exp = [exponential(t,epopt[0],epopt[1],epopt[2]) for t in list(range(day_now,future_day,1))] # do pred
        x_exp = list(range(day_now,future_day,1))
        y_exp = preds_exp
    
        if verbose:
            print('\n** Based on Exponential Fit **\n')
            print('\tR^2:', expr2)
            print('\tDoubling Time (represents overall growth): ', round(edoubletime,2), '(±', round(edoubletimeerror,2),') days')
            print('\n** Predicting day', future_day,'(',days,'days time)**\n')
            print('\tPredicted number of infections (exponential growth):',round(preds_exp[-1]))
    
    return exponentialworked, epopt, edoubletime, edoubletimeerror, expr2, preds_exp, x_exp, y_exp


def plotCasesandPredict(confirmed_cases_df, country, days, current_date, verbose=False, figs=False):
    # Get cases time series
    cases_time_series = get_country_cases_time_series(confirmed_cases_df, country)
    y = np.array(cases_time_series['Cases'])
    x = np.arange(y.size) 

    # Get double time for the last week
    if len(y) >= 7:
        current = y[-1]
        lastweek = y[-8]   
        ratio, dailypercentchange, recentdbltime = find_recent_doubling_time(current, lastweek)
        if verbose:
            print('\n** Based on Most Recent Week of Data **\n')
            print('\tConfirmed cases on',cases_time_series.index[-1],'\t',current)
            print('\tConfirmed cases on',cases_time_series.index[-8],'\t',lastweek)
            print('\tRatio:',round(ratio,2))
            print('\tWeekly increase:',round( 100 * (ratio - 1), 1),'%')
            print('\tDaily increase:', dailypercentchange, '% per day')
            print('\tDoubling Time (represents recent growth):',recentdbltime,'days')
    else:
        ratio, dailypercentchange, recentdbltime = (float('NaN'), float('NaN'), float('NaN'))
   

    # Find whether the curve is logistic or exponential
    logisticworked = False
    exponentialworked = False

    #TO-DO add which exception we are looking for here. We should not have naked try/excepts.
    try:
        logisticworked, lpopt, ldoubletime, ldoubletimeerror, logisticr2, preds_log, x_log, y_log = try_logistic(x, y, days, verbose)
    except:
        pass

    #TO-DO add which exception we are looking for here. We should not have naked try/excepts.
    try:
        exponentialworked, epopt, edoubletime, edoubletimeerror, expr2, preds_exp, x_exp, y_exp = try_exponential(x, y, days, verbose)
    except:
        pass

    if logisticworked and exponentialworked:
        if figs:
            logistic_graph(x, y, lpopt, x_log, y_log, country, current_date)
            exponential_graph(x, y, epopt, x_exp, y_exp, country, current_date)
        if round(logisticr2,2) > round(expr2,2):
            return [ldoubletime, ldoubletimeerror, recentdbltime, lpopt, round(preds_log[-1])]
        else:
            return [edoubletime, edoubletimeerror, recentdbltime, epopt, round(preds_exp[-1])]

    if logisticworked:
        if figs:
            logistic_graph(x, y, lpopt, x_log, y_log, country, current_date)
        return [ldoubletime, ldoubletimeerror, recentdbltime, lpopt, round(preds_log[-1])]

    if exponentialworked:
        if figs:
            exponential_graph(x, y, epopt, x_exp, y_exp, country, current_date)
        return [edoubletime, edoubletimeerror, recentdbltime, epopt, round(preds_exp[-1])]
    else:
        return [float('NaN'), float('NaN'), recentdbltime, float('NaN'), float('NaN')]


def main(days):
    # load data
    covid_confirmed_df, covid_deaths, covid_recovered = fetch_hopkins.load_model_data()
    # get most_recent_date
    dates = covid_confirmed_df.keys()
    current_date = dates[-1]
    # get the most recent cases_confirmed for each country
    cases = covid_confirmed_df.iloc[:,[1,-1]].groupby(COUNTRY_LABEL).sum().sort_values(by = current_date, ascending = False)
    # find highly affected countries
    topcountries = cases[cases[current_date] >= 100].index.tolist()
     
    timestamp = datetime.now().isoformat()

    total_results = {"results":[],"timestamp":timestamp}
    
    # import total medical capacity
    resources_capacity_dict = {}
    resources_capacity_df = pd.read_csv('../../data/external/country_medical_capacities.csv')
    for index, row in resources_capacity_df.iterrows():
        resources_capacity_dict[row['country_code']] = row['total_capacity']

    # collect country demands
    country_list = []
    demands_list = []

    # create full output dictionary
    for country in topcountries:
        # run prediction model
        dbltime, dbltimeerr, recentdbltime, params, pred = plotCasesandPredict(covid_confirmed_df, country, days, current_date)
        # initialise dict of results
        country_results_dict = {}
        
        ## Generate dict values
        # get country code
        country_code = country_code_dict[country]
        # get number of hospital beds
        if country_code in resources_capacity_dict:
            resources_capacity = resources_capacity_dict[country_code]
        else:
            resources_capacity = 0
        # get current corona cases
        confirmed = cases.loc[country][0]
        # get current corona deaths
        if(not covid_deaths.empty):
            deaths = int(covid_deaths[covid_deaths[COUNTRY_LABEL]==country].iloc[:,-1].sum())
        else:
            deaths = 0
        # get current corona recovered
        if(not covid_recovered.empty):
            recovered = int(covid_recovered[covid_recovered[COUNTRY_LABEL]==country].iloc[:,-1].sum())
        else:
            recovered = 0
        # get 3 week predicted cases
        if not np.isnan(pred):
            confirmed_prediction_3w = int(pred) # we take the prediction
        else:
            confirmed_prediction_3w = -1 # why -1, not 0?
        # get 3 week predicted deaths and recovered
        if confirmed != 0:
            deaths_prediction_3w = int(deaths/confirmed * confirmed_prediction_3w) # use current perc deaths
            recovered_prediction_3w = int(recovered/confirmed * confirmed_prediction_3w) # use current perc recovered
        else:
            deaths_prediction_3w = -1 # why -1, not 0?
            recovered_prediction_3w = -1 # why -1, not 0?

        # populate dict
        country_results_dict['country_code'] = country_code
        country_results_dict['country_name'] = country
        country_results_dict['resources_capacity'] = int(resources_capacity)
        country_results_dict['covid19_capacity'] = int(round(resources_capacity * 0.05 * 0.5)) # <- 5% of hospital beds are ICD and HDU, 50% are filled with normal patients
        country_results_dict['confirmed'] = int(confirmed)
        country_results_dict['deaths'] = int(deaths)
        country_results_dict['recovered'] = int(recovered)
        country_results_dict['confirmed_prediction_3w'] = int(confirmed_prediction_3w)
        country_results_dict['deaths_prediction_3w'] = int(deaths_prediction_3w)
        country_results_dict['recovered_prediction_3w'] = int(recovered_prediction_3w)
        country_results_dict['resource_requirements_current'] = int(round((confirmed - deaths - recovered) * 0.15))
        country_results_dict['resource_requirements_3w'] = int(round((confirmed_prediction_3w - deaths_prediction_3w - recovered_prediction_3w) * 0.15))

        # append to master dict
        total_results['results'].append(country_results_dict)

        # append requirements/excess to demand list for the demands file -> distribution script
        demand = country_results_dict['covid19_capacity'] - country_results_dict['resource_requirements_3w']
        demands_list.append(demand)
        country_list.append(country_code)

    demands_frame = pd.DataFrame({'country':country_list, 'demand':demands_list})
    demands_frame.to_csv("../distribution/demands.csv", index=False)

    return total_results


if __name__ == "__main__":
    days = 21 # 3 weeks prediction
    results = main(days)
    # print(results)
    populate_results.populate_with_predicted_cases(results)
