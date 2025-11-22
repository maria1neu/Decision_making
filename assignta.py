"""
Gabriella Montalvo & Maria Samos Rivas
DS3500 Homework 6: Resource Allocation with Evolutionary Computing
Due 11/23/25
assignta.py
"""
import pandas as pd
import random as rnd
import numpy as np
from evo import Evo

# OBJECTIVE 1
def overallocation(sol):
    """
    Parameters:
    Returns:
    Does:
    """
    assignments = sol['assignments']
    tas = sol['tas']

    # Counts how many sections a TA is assigned
    assigned_section_count = np.sum(assignments, axis = 1)

    # Gets the max number of allowed assignments for each TA
    max_allowed = tas['max_assigned'].values

    # Calculates the number of penalties, using vector math so negatives become 0
    penalties = np.maximum(assigned_section_count - max_allowed, 0)
    total_penalties = np.sum(penalties)

    return total_penalties

# OBJECTIVE 2
def conflicts(sol):
    """
    Parameters:
    Returns:
    Does:
    """
    assignments = sol['assignments']
    sections = sol['sections']

    time_conflicts_count = 0

    # Loops over all TAs and their assignments
    for i in range(assignments.shape[0]):
        sections_assigned = np.where(assignments[i] == 1)[0]

        # Collects all the meeting times for each lab section
        section_times = sections.loc[sections_assigned, 'daytime'].values


        # If length of unique times is less than the number of assigned sections, conflict exists!
        if len(np.unique(section_times)) < len(section_times):
            time_conflicts_count += 1

    return time_conflicts_count

# OBJECTIVE 3
def undersupport(sol):
    """
    Parameters:
    Return:
    Does:
    """

    assignments = sol['assignments']
    sections = sol['sections']

    assigned_section_count = np.sum(assignments, axis = 0)

    # collects all values for the minimum number of tas per section
    min_ta = sections['min_ta']

    #  Calculates the number of penalties, using vector math so negatives become 0
    penalties = np.maximum(min_ta - assigned_section_count, 0)
    total_penalties = np.sum(penalties)

    return total_penalties

#OBJECTIVE 4
def unavailable(sol):
    """
    Parameters:
    Returns:
    Does:
    """
    tas = sol['tas']
    assignments = sol['assignments']

    penalty = 0

    # loops through all the columns and rows to see ta availability
    for i in range(assignments.shape[0]):
        for j in range(assignments.shape[1]):
            # adds a penalty if the ta was scheduled but wrote unavailable
            if assignments[i][j] == 1 and tas.iloc[i, j + 3] == 'U':
                penalty += 1
            else:
                penalty = penalty

    return penalty

def unpreferred(sol):
    """
    Parameters:
    Returns:
    Does:
    """

    tas = sol['tas']
    assignments = sol['assignments']

    penalty = 0

    for i in range(assignments.shape[0]):
        for j in range(assignments.shape[1]):
            if assignments[i][j] == 1 and tas.iloc[i, j + 3] == 'W':
                penalty += 1

    return penalty

#randomized initial solution
def inital_solutions():

    tas = pd.read_csv("tas.csv")
    sections = pd.read_csv("sections.csv")

    avail_matrix = tas.iloc[:, 3:].to_numpy()
    max_assigned_ta = tas["max_assigned"].to_numpy()

    num_tas, num_sections = avail_matrix.shape
    assignments = np.zeros((num_tas, num_sections), dtype = int)

    for i in range(num_tas):
        allowed_sections = [j for j in range(num_sections) if avail_matrix[i, j] != 'U']
        k = rnd.randint(0, max_assigned_ta[i])
        k = min(k, len(allowed_sections))

        chosen_sections = rnd.sample(allowed_sections, k)

        for j in chosen_sections:
            assignments[i, j] = 1

    for j in range(num_sections):
        current_tas = assignments[:, j].sum()
        required_tas = sections['min_ta'].iloc[j]

        while current_tas < required_tas:
            candidates = [
                i for i in range(num_tas)
                if avail_matrix[i, j] != "U"
                   and assignments[i, j] == 0
                   and assignments[i, :].sum() < max_assigned_ta[i]
            ]
            if len(candidates) == 0:
                break

            i = rnd.choice(candidates)
            assignments[i, j] = 1
            current_tas += 1

    return { "assignments": assignments, "tas": tas, "sections": sections }

def agent_fix_unavailable(parents):

    sol = parents[0]
    tas = sol['tas']
    assignments = sol['assignments']

    avail_matrix = tas.iloc[:, 3:].to_numpy()
    num_tas, num_sections = assignments.shape

    i = rnd.randint(0, num_tas - 1)
    j = rnd.randint(0, num_sections - 1)

    if assignments[i, j] == 1 and avail_matrix[i, j] == 'U':
        assignments[i, j] = 0

    return sol

def agent_fix_unpreferred(parents):

    sol = parents[0]
    tas = sol['tas']
    assignments = sol['assignments']

    avail_matrix = tas.iloc[:, 3:].to_numpy()
    num_tas, num_sections = assignments.shape

    i = rnd.randint(0, num_tas - 1)
    j = rnd.randint(0, num_sections - 1)

    if assignments[i, j] == 1 and avail_matrix[i, j] == 'W':
        assignments[i, j] = 0

    return sol

def agent_random_flip(parents):

    sol = parents[0]
    assignments = sol["assignments"]

    n_ta, n_sec = assignments.shape
    i = rnd.randrange(0, n_ta)
    j = rnd.randrange(0, n_sec)

    assignments[i, j] = 1 - assignments[i, j]

    return sol

def agent_reduce_overallocation(parents):

    sol = parents[0]
    tas = sol["tas"]
    assignments = sol["assignments"]

    max_assigned_ta = tas["max_assigned"].to_numpy()

    num_tas, num_sections = assignments.shape

    i = rnd.randint(0, num_tas - 1)
    assigned_sections = [j for j in range(num_sections) if assignments[i, j] == 1]

    if len(assigned_sections) > max_assigned_ta[i]:
        j = rnd.choice(assigned_sections)
        assignments[i, j] = 0

    return sol

# Adding the objectives to evo!
evo = Evo()
evo.add_objective('overallocation', overallocation)
evo.add_objective('conflicts', conflicts)
evo.add_objective('undersupport', undersupport)
evo.add_objective('unavailable', unavailable)
evo.add_objective('unpreferred', unpreferred)

for _ in range(20):
    evo.add_solution(inital_solutions())

# Register agents
evo.add_agent("fix_unavailable", agent_fix_unavailable, k=1)
evo.add_agent("fix_unpreferred", agent_fix_unpreferred, k=1)
evo.add_agent("random_flip", agent_random_flip, k=1)
evo.add_agent("reduce_overallocation", agent_reduce_overallocation, k=1)

print("Initial nondominated population:")
print(evo)

# Run for 5 minutes
evo.evolve(time_limit=300)

print("\nFinal nondominated population:")
print(evo)