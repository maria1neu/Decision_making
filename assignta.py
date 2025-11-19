"""
Gabriella Montalvo & Maria Samos Rivas
DS3500 Homework 6: Resource Allocation with Evolutionary Computing
Due 11/23/25
assignta.py
"""
import pandas as pd
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

    min_ta = sections['min_ta']

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

    for i in range(assignments.shape[0]):
        for j in range(assignments.shape[1]):
            if assignments[i][j] == 1 and tas.iloc[i, j + 3] == 'U':
                penalty += 1
            else:
                penalty = penalty

    return penalty

# Adding the objectives to evo!
evo = Evo()
evo.add_objective('overallocation', overallocation)
evo.add_objective('conflicts', conflicts)
evo.add_objective('undersupport', undersupport)
evo.add_objective('unavailable', unavailable)