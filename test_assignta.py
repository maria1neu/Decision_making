"""
Gabriella Montalvo & Maria Samos Rivas
DS3500 Homework 6: Resource Allocation with Evolutionary Computing
Due 11/23/25
test_assignta.py
"""
import pytest
import numpy as np
import pandas as pd
from assignta import overallocation, conflicts, undersupport


# Loading in the tests from the csv files
def get_test_sol(num):
    """
    Parameters:
    Returns:
    Does:
    """
    tas = pd.read_csv('tas.csv')
    sections = pd.read_csv('sections.csv')
    assignments = np.loadtxt(f'test{num}.csv', delimiter = ',')

    test_solutions = {
        'assignments': assignments,
        'tas': tas,
        'sections': sections
    }

    return test_solutions

# Test for Objective 1
def test_overallocation():
    sol1 = get_test_sol(1)
    sol2 = get_test_sol(2)
    sol3 = get_test_sol(3)

    assert overallocation(sol1) == 34
    assert overallocation(sol2) == 37
    assert overallocation(sol3) == 19
# PASSED

# Test for Objective 2
def test_conflicts():
    sol1 = get_test_sol(1)
    sol2 = get_test_sol(2)
    sol3 = get_test_sol(3)

    assert conflicts(sol1) == 7
    assert conflicts(sol2) == 5
    assert conflicts(sol3) == 2
# PASSED

# Test for Objective 3
def test_undersupport():
    sol1 = get_test_sol(1)
    sol2 = get_test_sol(2)
    sol3 = get_test_sol(3)

    assert undersupport(sol1) == 1
    assert undersupport(sol2) == 0
    assert undersupport(sol3) == 11
