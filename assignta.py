"""
Gabriella Montalvo & Maria Samos Rivas
DS3500 Homework 6: Resource Allocation with Evolutionary Computing
Due 11/23/25
assignta.py
"""

# importing external libraries for later use
import pandas as pd
import random as rnd
import numpy as np
from evo import Evo
from profiler import profile
from profiler import Profiler

# OBJECTIVE 1
@profile
def overallocation(sol):
    """
    Parameters: solution (lst) with a dictionary inside
    Returns: integer
    Does: counts the amount of times TAs are assigned more than they would like
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
@profile
def conflicts(sol):
    """
    Parameters: solution (lst) with a dictionary inside
    Returns: integer
    Does: counts the amount of times theres time conflicts in the assignments
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
@profile
def undersupport(sol):
    """
    Parameters: solution (lst) with a dictionary inside
    Return: integer
    Does: counts the times there aren't enough TAs
    """

    # gets data
    assignments = sol['assignments']
    sections = sol['sections']

    # counts up the tas per section
    assigned_section_count = np.sum(assignments, axis = 0)

    # collects all values for the minimum number of tas per section
    min_ta = sections['min_ta']

    # Calculates the number of penalties, using vector math so negatives become 0
    penalties = np.maximum(min_ta - assigned_section_count, 0)
    total_penalties = np.sum(penalties)

    return total_penalties

# OBJECTIVE 4
@profile
def unavailable(sol):
    """
    Parameters: solution (lst) with a dictionary inside
    Returns: integer
    Does: count the number of times you allocate a TA to a section they are unavailable to support
    """

    # gets data
    assignments = sol['assignments']
    tas = sol['tas']

    # gets the availability column for tas
    availability_columns = [str(i) for i in range(assignments.shape[1])]
    availability = tas[availability_columns].to_numpy()

    # Checking the Unavailability of the TAs and counts if its unavailable
    penalty = np.sum((assignments == 1) & (availability == 'U'))

    return penalty

# OBJECTIVE 5
@profile
def unpreferred(sol):
    """
    Parameters: solution (lst) with a dictionary inside
    Returns: integer
    Does: Count the number of times you allocate a TA to a section where they said “willing” but not
“preferred”
    """
    assignments = sol['assignments']
    availability = sol['tas'][[str(i) for i in range(assignments.shape[1])]].to_numpy()

    # Checking if TAs preference is WILLING, a change from the previous objective
    preference = np.sum((assignments == 1) & (availability == 'W'))

    return preference

# randomized initial solution
def inital_solutions():


    # gets data
    tas = pd.read_csv("tas.csv")
    sections = pd.read_csv("sections.csv")

    # Gets the TA availability matrix
    avail_matrix = tas.iloc[:, 3:].to_numpy()
    max_assigned_ta = tas["max_assigned"].to_numpy()

    # gets the tas and sections and sets everything to zero
    num_tas, num_sections = avail_matrix.shape
    assignments = np.zeros((num_tas, num_sections), dtype = int)

    # assign each TA a random set of sections they're allowed to take
    for i in range(num_tas):
        # randomly picks TA's where there are available
        allowed_sections = [j for j in range(num_sections) if avail_matrix[i, j] != 'U']
        k = rnd.randint(0, max_assigned_ta[i]) # doesn't exceed their max sessions
        k = min(k, len(allowed_sections))

        # randomly pick k allowed sections for this TA
        chosen_sections = rnd.sample(allowed_sections, k)

        # addigns TA to sections
        for j in chosen_sections:
            assignments[i, j] = 1

    # make sure every section has enough TAs
    for j in range(num_sections):
        #finds number of ta currently assigned to the section and the min requirement
        current_tas = assignments[:, j].sum()
        required_tas = sections['min_ta'].iloc[j]

        # If we're below the minimum, keep adding TAs until we reach it
        while current_tas < required_tas:
            candidates = [
                i for i in range(num_tas)
                if avail_matrix[i, j] != "U"
                   and assignments[i, j] == 0
                   and assignments[i, :].sum() < max_assigned_ta[i]
            ]
            # if no candidate exists, we can't fix this section
            if len(candidates) == 0:
                break

            # randomly add one of the valid TAs
            i = rnd.choice(candidates)
            assignments[i, j] = 1
            current_tas += 1

    # returns new solution dictionary
    return { "assignments": assignments, "tas": tas, "sections": sections }

# AGENT 1
@profile
def agent_fix_unavailable(parents):
    """
    Parameters: original solution (lst) containing one dictionary
    Returns: modified solution (dict)
    Does: picks a random TA and if they have an unavailable 'U' part it removes assignment
    """

    # access data
    sol = parents[0]
    tas = sol['tas']
    assignments = sol['assignments']

    # coverts availability to numpy array and gets number of TAs and Sections
    avail_matrix = tas.iloc[:, 3:].to_numpy()
    num_tas, num_sections = assignments.shape

    # picks random TA (row) and section(column)
    i = rnd.randint(0, num_tas - 1)
    j = rnd.randint(0, num_sections - 1)

    # is TA is picked and unavailable --> removes assignment
    if assignments[i, j] == 1 and avail_matrix[i, j] == 'U':
        assignments[i, j] = 0

    return sol

# AGENT 2
@profile
def agent_fix_unpreferred(parents):
    """
    Parameters: original solution (dict)
    Returns: modified solution (dict)
    Does: picks a random TA and if they have an unpreferred 'W' part it removes assignment
    """

    # access data
    sol = parents[0]
    tas = sol['tas']
    assignments = sol['assignments']

    # coverts availability to numpy array and gets number of TAs and Sections
    avail_matrix = tas.iloc[:, 3:].to_numpy()
    num_tas, num_sections = assignments.shape

    # picks random TA (row) and section(column)
    i = rnd.randint(0, num_tas - 1)
    j = rnd.randint(0, num_sections - 1)

    # is TA is picked and not preferred --> removes assignment
    if assignments[i, j] == 1 and avail_matrix[i, j] == 'W':
        assignments[i, j] = 0

    return sol

# AGENT 3
@profile
def agent_random_flip(parents):
    """
    Parameters: original solution (dict)
    Returns: modified solution (dict)
    Does: randomly picks a TA and a section and flips the assignment
    """

    # access data
    sol = parents[0]
    assignments = sol["assignments"]

    # gets number of TAs and Sections and randomly assigns them
    n_ta, n_sec = assignments.shape
    i = rnd.randrange(0, n_ta)
    j = rnd.randrange(0, n_sec)

    # gets the ranmodly assigned ta and section and flips by changing assignment from 1 to 0 or 0 to 1
    assignments[i, j] = 1 - assignments[i, j]

    return sol

# AGENT 4
@profile
def agent_reduce_overallocation(parents):
    """
    Parameters: original solution (dict)
    Returns: modified solution (dict)
    Does: removes TA's who are assigned to too many sections
    """

    # access data
    sol = parents[0]
    tas = sol["tas"]
    assignments = sol["assignments"]

    # gets the max for tas assignment
    max_assigned_ta = tas["max_assigned"].to_numpy()

    # gets the tas and sections
    num_tas, num_sections = assignments.shape

    # rabdomly picks a ta and gets the number of assigned sections
    i = rnd.randint(0, num_tas - 1)
    assigned_sections = [j for j in range(num_sections) if assignments[i, j] == 1]

    # if assigned sections exceeds their max, they get their assignments changed
    if len(assigned_sections) > max_assigned_ta[i]:
        j = rnd.choice(assigned_sections)
        assignments[i, j] = 0

    return sol

if __name__ == "__main__":
    # Adding the objectives to evo!
    evo = Evo()
    evo.add_objective('overallocation', overallocation)
    evo.add_objective('conflicts', conflicts)
    evo.add_objective('undersupport', undersupport)
    evo.add_objective('unavailable', unavailable)
    evo.add_objective('unpreferred', unpreferred)
    for _ in range(5):
        evo.add_solution(inital_solutions())
    # Register agents
    evo.add_agent("fix_unavailable", agent_fix_unavailable, k=1)
    evo.add_agent("fix_unpreferred", agent_fix_unpreferred, k=1)
    evo.add_agent("random_flip", agent_random_flip, k=1)
    evo.add_agent("reduce_overallocation", agent_reduce_overallocation, k=1)
    # Run for 5 minutes
    evo.evolve(time_limit=300)

    # Run summary table!
    evo.summarize('gabymari_summary.csv', groupname = 'gabymari')

    # profile report for gabymari_profile.txt
    Profiler.report()
