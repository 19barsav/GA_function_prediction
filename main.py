import random
import math
import numpy as np
import sys
from copy import deepcopy

"""
Attempts to find the target function f(x) = x^3 - 2x^2 + x using genetic programming.
Currently not well organized or commented - 
    but the visualization and algorithm work great!
"""

def print_tree(root, level=0, prefix="Root: "):
    if root:
        print(" " * (level * 4) + prefix + str(root.value) + "  " + str(root.deep))
        if root.left or root.right:
            print_tree(root.left, level + 1, "L--- ")
            print_tree(root.right, level + 1, "R--- ")


# Define the target function f(x)
def target_function(x):
    return x ** 3 - 2 * x ** 2 + x


x_range = [x / 10 for x in range(-50, 51)]
population_size = 100
num_generations = 100
tournament_size = 3
mutation_rate = 0.2
max_depth = 5


# Define the TreeNode class and random tree generation function
class TreeNode:
    def __init__(self, value, parent=None, deep=0):
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None
        self.deep = deep


def generate_random_tree(depth):
    if depth <= 0 or (random.uniform(0, 1) <= 0.1 and depth < 2):
        value = random.choice(["x", "const"])
        if value == "const":
            value = random.choice(["1.0", "2.0", "3.0"])
    else:
        value = random.choice(["+", "-", "*", "/"])

    root = TreeNode(value, deep=depth)

    if value in ["+", "-", "*", "/"]:
        root.left = generate_random_tree(depth - 1)
        if root.left:
            root.left.parent = root
        root.right = generate_random_tree(depth - 1)
        if root.right:
            root.right.parent = root

    return root


# Define the fitness function
def fitness(program, x_values, target_func):
    error = 0
    for x in x_values:
        result = evaluate_tree(program, x)
        error += (target_func(x) - result) ** 2

    return error  # Inverse of squared error, with a small epsilon to prevent division by zero


# Define a function to evaluate a tree
def evaluate_tree(node, x):
    if node.value == "x":
        return x
    if node.value == "1.0":
        return 1.0
    if node.value == "2.0":
        return 2.0
    if node.value == "3.0":
        return 3.0

    left_value = evaluate_tree(node.left, x)
    right_value = evaluate_tree(node.right, x)

    if node.value == "+":
        return left_value + right_value
    elif node.value == "-":
        return left_value - right_value
    elif node.value == "*":
        return left_value * right_value
    elif node.value == "/":
        return left_value / right_value if right_value != 0 else 0


# Define the tournament selection function
def tournament_selection(population, fitness_values, tournament_size):
    selected_indices = []
    for x in range(tournament_size):
        random_index = random.randint(0, len(population) - 1)
        selected_indices.append(random_index)
    best_index = min(selected_indices, key=lambda i: fitness_values[i])
    return population[best_index]


# Define the subtree crossover function
def subtree_crossover(parent1, parent2):
    node_to_cross1 = deepcopy(parent1)
    node_to_cross2 = deepcopy(parent2)

    for i in range(1, random.randint(1, max([parent1.deep, parent2.deep]) - 1)):
        side = random.choice(["left", "right"])
        if side == "left" and node_to_cross1.left and node_to_cross2.left:
            node_to_cross1.deep, node_to_cross2.deep = node_to_cross2.deep, node_to_cross1.deep
            node_to_cross1 = node_to_cross1.left
            node_to_cross2 = node_to_cross2.left
        elif side == "right" and node_to_cross1.right and node_to_cross2.right:
            node_to_cross1.deep, node_to_cross2.deep = node_to_cross2.deep, node_to_cross1.deep
            node_to_cross1 = node_to_cross1.right
            node_to_cross2 = node_to_cross2.right
        elif side == "left" and node_to_cross1.left and node_to_cross2.right:
            node_to_cross1.deep, node_to_cross2.deep = node_to_cross2.deep, node_to_cross1.deep
            node_to_cross1 = node_to_cross1.left
            node_to_cross2 = node_to_cross2.right
        elif side == "right" and node_to_cross1.right and node_to_cross2.left:
            node_to_cross1.deep, node_to_cross2.deep = node_to_cross2.deep, node_to_cross1.deep
            node_to_cross1 = node_to_cross1.right
            node_to_cross2 = node_to_cross2.left
        else:
            break
    if not node_to_cross1.parent or not node_to_cross2.parent:
        return node_to_cross1, node_to_cross2
    if node_to_cross1.parent.left and node_to_cross1.parent.left == node_to_cross1:
        node_to_cross1.parent.left = node_to_cross2
    else:
        node_to_cross1.parent.right = node_to_cross2

    if node_to_cross2.parent.left and node_to_cross2.parent.left == node_to_cross2:
        node_to_cross2.parent.left = node_to_cross1
    else:
        node_to_cross2.parent.right = node_to_cross1

    node_to_cross1.parent, node_to_cross2.parent = node_to_cross2.parent, node_to_cross1.parent

    while node_to_cross1.parent != None:
        node_to_cross1 = node_to_cross1.parent
    while node_to_cross2.parent != None:
        node_to_cross2 = node_to_cross2.parent
    return node_to_cross1, node_to_cross2


# Define the subtree mutation function
def subtree_mutation(program, mutation_rate):
    if random.random() < mutation_rate:
        mutation_point = random.choice(find_operator_nodes(program))
        mutation_point.value = random.choice(["+", "-", "*", "/"])
        if mutation_point.left:
            mutation_point.left = generate_random_tree(mutation_point.deep - 1)
            mutation_point.left.parent = mutation_point

        if mutation_point.right:
            mutation_point.right = generate_random_tree(mutation_point.deep - 1)
            mutation_point.right.parent = mutation_point


# Find operator nodes in the tree
def find_operator_nodes(node):
    operator_nodes = []
    if node.value in ["+", "-", "*", "/"]:
        operator_nodes.append(node)
    if operator_nodes:
        side = random.choice(["right", "left", "stop"])
        if side == "left" and node.left:
            operator_nodes.extend(find_operator_nodes(node.left))
        elif side == "right" and node.right:
            operator_nodes.extend(find_operator_nodes(node.right))
        else:
            return operator_nodes
    else:
        if node.left:
            operator_nodes.extend(find_operator_nodes(node.left))
        if node.right:
            operator_nodes.extend(find_operator_nodes(node.right))
    return operator_nodes


def genetic_programming(target_func, x_values, pop_size, max_generations, tournament_size, mutation_rate):
    # Initialize the population
    best_programs = []
    population = []
    for i in range(population_size):
        x = random.randint(2, max_depth)
        population.append(generate_random_tree(x))

    # Main GP loop
    for generation in range(num_generations):
        fitness_values = [fitness(program, x_range, target_function) for program in population]
        if generation % 10 == 0:
            best_fit = min(fitness_values)
            best_index = fitness_values.index(best_fit)
            best_program = population[best_index]
            print_tree(best_program)
            best_programs.append(best_program)

        # Print the best program in this generation
        print(f"Generation {generation + 1} - Best Fitness: {min(fitness_values)}")
        # Tournament selection and crossover
        new_population = []
        for i in range(population_size):
            parent1 = tournament_selection(population, fitness_values, 10)
            parent2 = tournament_selection(population, fitness_values, 10)
            # FIX THIS
            child1, child2 = subtree_crossover(parent1, parent2)
            subtree_mutation(child1, mutation_rate)
            subtree_mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    return population, best_programs


final_population, best_programs = genetic_programming(target_function, x_range, population_size, num_generations,
                                                      tournament_size, mutation_rate)
print(len(final_population))
fitness_values = [fitness(program, x_range, target_function) for program in final_population]
best_program = final_population[fitness_values.index(min(fitness_values))]

print("Final Best Program:")
print_tree(best_program)

import matplotlib.pyplot as plt


def plot_program_evolution(best_program, generations):
    plt.figure(figsize=(10, 5))
    plt.plot(x_range, [target_function(x) for x in x_range], label="Target Function")
    for i, best_program in enumerate(best_programs):
        y_values = [evaluate_tree(best_program, x) for x in x_range]
        plt.plot(x_range, y_values, label=f"Evolved Program {i}")

    plt.legend()
    plt.ticklabel_format(style='plain')
    plt.title(f"Evolution of Best Program over {generations} Generations")
    plt.show()


plot_program_evolution(best_program, num_generations)