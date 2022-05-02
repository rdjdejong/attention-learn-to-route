#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import argparse
from itertools import combinations, permutations, cycle, islice
import numpy as np
from utils.data_utils import load_dataset, save_dataset
from gurobipy import *


def solve_euclidian_tsp_dynamic(points, revealed, start=None, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate
    :return:
    """

    n = revealed[-1]

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys()
                                 if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in permutations(tour, 2))
                             <= len(tour) - 1)
            elif viable_tour(selected) is None:
                model.cbLazy(quicksum(model._vars[i, j]
                                        for i, j in selected) <= n - 2)
                model.cbLazy(quicksum(model._vars[j, i]
                                        for i, j in selected) <= n - 2)
            else:
                pass #print(viable_tour(selected))



    # Given a tuplelist of edges, find the shortest subtour
    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Check whether if we follow this tour from any point is a viable tour
    def viable_tour(edges):
        edges.sort(key=lambda y: y[0])
        for idx in range(n):
            tour = viable_tour_(edges, idx)
            if len(tour) == n:
                return tour

        edges.sort(key=lambda y: y[1])
        for idx in range(n):
            tour = viable_tour_(edges, idx, reverse=True)
            if len(tour) == n:
                return tour
        return None


    def viable_tour_(edges, idx, reverse=False):
        tour = []
        for t in range(1, n):
            if reverse:
                check = (edges[idx][0] <= revealed[t] and
                         edges[idx][1] < revealed[t])
            else:
                check = (edges[idx][0] < revealed[t] and
                         edges[idx][1] <= revealed[t])

            if not check:
                return tour

            tour.append(idx)
            if reverse:
                idx = edges[idx][0]
            else:
                idx = edges[idx][1]

        tour.append(idx)
        return tour

    # Dictionary of Euclidean distance between each pair of points
    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(n) if i != j}

    m = Model()

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint
    for i in range(n):
        m.addConstr(sum(vars[i,j] for j in range(n) if i != j) == 1)
        m.addConstr(sum(vars[j,i] for j in range(n) if i != j) == 1)

        if start is None:
            if i < n-1:
                vars[i, i+1].Start = 1.0
            else:
                vars[i, 0].Start = 1.0

    if start is not None:
        for i,j in start:
            vars[i, j].Start = 1.0

    # Optimize model
    m._vars = vars
    m.Params.outputFlag = False
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = viable_tour(selected)
    assert len(tour) == n

    return m.objVal, tour


def solve_euclidian_tsp(points, tour_taken, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate
    :return:
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    if len(tour_taken) > 1:
        fixed_tour = list(zip(tour_taken, tour_taken[1:]))
        for i,j in vars.keys():
            if i in tour_taken and (i, j) in fixed_tour:
                    m.addConstr(vars[i, j] == 1)

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour


def solve_all_gurobi(dataset):
    results = []
    for i, instance in enumerate(dataset):
        print ("Solving instance {}".format(i))
        result = solve_euclidian_tsp(instance)
        results.append(result)
    return results
