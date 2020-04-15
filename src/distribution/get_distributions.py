'''
This file takes as an input a dataframe with the nominal costs of transportation
between different countries, plus a dictionary of the countries that have either
an excess or decifiency in medical supplies. It runs a naive approach at
optimising the districution of resources, and will return a dictionary showing
which countries should make transfers to which. 

'''

import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, '../')
from utils import populate_results
from datetime import datetime


def make_graph(nodes, edges):
    '''
    Makes the networkx graph with all the countries. Edge-weight is the cost of
    transport between the two countries.
    '''
    graph = nx.MultiGraph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return graph


def country_preferred_transaction(subgraph, country, selling_nodes):
    '''
    Finds the cheapest available transaction for a given country
    '''
    edges_of_interest = list(set(list(subgraph.edges(country, 'weight'))))
    edges_of_interest = [edge for edge in edges_of_interest if edge[1] in selling_nodes]
    return sorted(edges_of_interest, key= lambda x: x[2])[0]


def find_next_transaction(requirements_dict, graph, recognised_countries):
    '''
    Find the optimal next transaction to make, from remaining active nodes. 
    '''
    unrecognised_countries = 0

    # Filter the graph to the transacting nodes
    active_nodes = list(requirements_dict.keys())
    subgraph = graph.subgraph(active_nodes)
    
    # Split the transacting nodes into buyers and sellers
    buying_nodes = [country for country, capacity in requirements_dict.items() if capacity < 0]
    selling_nodes = [country for country, capacity in requirements_dict.items() if capacity > 0]
    # Find the cheapest transaction for each country
    country_preferred_transactions = []
    for country in buying_nodes:
        if country in recognised_countries:
            country_preferred_transactions.append(country_preferred_transaction(subgraph, country, selling_nodes))
        else:
            unrecognised_countries+=1

    # Find the cheapest transaction overall
    if country_preferred_transactions:
        preferred_transactions_sorted = sorted(country_preferred_transactions, key= lambda x: x[2])
    # If no country left in the buying list is recognised, trigger ending the distribution optimisation
    elif len(buying_nodes)==unrecognised_countries:
        return False
    # Take all transactions at the lowest cost
    min_cost = preferred_transactions_sorted[0][2]
    lowest_cost_transactions = [transaction for transaction in preferred_transactions_sorted if transaction[2]==min_cost]
    
    # Add their requirements
    transactions_with_health_values = []
    for transaction in lowest_cost_transactions:
        transaction_dict = {} # turn remaining transactions into dicts for more intuitive use
        transaction_dict['recipient'] = transaction[0]
        transaction_dict['donor'] = transaction[1]
        transaction_dict['required_value'] = requirements_dict[transaction[0]]
        transaction_dict['incoming_value'] = requirements_dict[transaction[1]]
        transaction_dict['cost'] = transaction[2]
        transactions_with_health_values.append(transaction_dict)

    selected_transaction = sorted(transactions_with_health_values, key= lambda x: x['required_value'])[0]
    
    return selected_transaction


def update_requirements_dictionary(selected_transaction, requirements_dictionary):
    '''
    Changes the values in the requirements dictionary to reflect that the trans
    action has been made.
    '''
    
    recipient = selected_transaction['recipient']
    donor = selected_transaction['donor']
    required_value = selected_transaction['required_value']
    incoming_value = selected_transaction['incoming_value']
    
    if abs(required_value) > incoming_value:
        # More needed!
        requirements_dictionary[recipient] += incoming_value
        del requirements_dictionary[donor]
    
    elif abs(required_value) < incoming_value:
        # Too much!
        del requirements_dictionary[recipient]
        requirements_dictionary[donor] += required_value # + because this is a negative number

    elif abs(required_value) == incoming_value:
        # Just right!
        del requirements_dictionary[donor]
        del requirements_dictionary[recipient]


def find_optimal_transactions(costs_df_location, requirements_df_location):
    '''
    Main function. Takes the costs and requirements files as inputs and returns
    a naively optimised list of transactions. 
    '''
    transactions = []

    costs = pd.read_csv(costs_df_location, keep_default_na=False)
    recognised_countries = costs['base_country'].unique()
    requirements = pd.read_csv(requirements_df_location)
    
    requirements_dictionary = {}
    for index, row in requirements.iterrows():
        requirements_dictionary[row['country']] = row['demand']  

    # Add a buffer, so that countries don't run out of resources completely
    for country, capacity in requirements_dictionary.items():
        if (capacity <= 150) and (capacity >= 0):
            requirements_dictionary[country] = 0
        elif capacity > 150:
            requirements_dictionary[country] = capacity - 150
    
    # Create the main graph
    nodes = costs['base_country'].unique()
    weighted_edges = []
    for index, row in costs.iterrows():
        we = (row['base_country'], row['to_country'], row['transaction_cost'])
        weighted_edges.append(we)
    
    G = make_graph(nodes, weighted_edges)
    
    # Run the transactions
    # While there are positive and negative numbers in the requirements_dictionary...
    while ((any(capacity > 0 for capacity in requirements_dictionary.values())) and
           (any(capacity < 0 for capacity in requirements_dictionary.values()))):
        selected_transaction = find_next_transaction(requirements_dictionary, G, recognised_countries)
        if selected_transaction == False: # if only unrecognised countries are left
            break
        selected_transaction['transfer_amount'] = min(abs(selected_transaction['required_value']), selected_transaction['incoming_value'])
        
        update_requirements_dictionary(selected_transaction, requirements_dictionary)

        del selected_transaction['required_value']
        del selected_transaction['incoming_value']
            
        transactions.append(selected_transaction)
    
    return transactions


transactions = find_optimal_transactions('country_distances.csv', 'demands.csv')
now = datetime.now()
distributions = {"timestamp": now, "distributions": transactions}
result = populate_results.populate_with_distributions(distributions)

# print(result)
# print(transactions)