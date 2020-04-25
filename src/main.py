import sys

from model import model_growth
from distribution import get_distributions
from utils import populate_results
from datetime import datetime

do_show_results = False
if (len (sys . argv) > 1) :
  if (sys . argv [1] == '--debug') :
    do_show_results = True

if __name__ == "__main__":
    # Get predictions
    days = 21 # 3 weeks prediction
    results = model_growth.main(days, 
                                resource_capacity_location='../data/external/country_medical_capacities.csv', 
                                demands_output_location='distribution/demands.csv',
                                do_show_results = do_show_results)
    # print(results)
    populate_results.populate_with_predicted_cases(results)

    # Get transactions
    transactions = get_distributions.find_optimal_transactions(costs_df_location='../data/external/country_distances.csv', 
                                                               requirements_df_location='distribution/demands.csv')
    now = datetime.now()
    distributions = {"timestamp": now, "distributions": transactions}
    # print(distributions)
    result = populate_results.populate_with_distributions(distributions)
