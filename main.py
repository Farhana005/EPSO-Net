import os
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from epso_nas import search_space, random_architecture, evaluate_fitness, crossover_and_mutate
from model import best_model
from dataloader import DataGenerator1, DataGenerator2

def main():
    TRAIN_DATASET_PATH = "E:/datasets/BraTS2021_Training_Data/"
    case_dirs = [f for f in os.listdir(TRAIN_DATASET_PATH) if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, f))]
    case_dirs = sorted(case_dirs)
    train_ids, val_ids = train_test_split(case_dirs, test_size=0.2, random_state=42)

    training_generator = DataGenerator1(train_ids, batch_size=8, n_channels=2)
    valid_generator = DataGenerator2(val_ids, batch_size=8, n_channels=2)

    population_size = 5
    num_generations = 10
    mutation_rate = 0.1
    num_epochs = 100
    n_channels = 2
    input_shape = (128, 128, 128, n_channels)
    n_classes = 3

    global_best_fitness = -np.inf
    global_best_individual = None
    global_best_metrics = None
    log = []

    def select_best_individuals(population, fitness_scores, num_selections):
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        return sorted_pop[:num_selections]

    population = []
    for _ in range(population_size):
        individual = random_architecture(search_space)
        individual["mutation_rate"] = mutation_rate
        population.append(individual)

    for generation in range(num_generations):
        fitness_scores = []
        for individual_index, individual in enumerate(population):
            fitness, metrics = evaluate_fitness(
                individual,
                training_generator,
                valid_generator,
                num_epochs,
                n_channels,
                generation,
                individual_index
            )
            fitness_scores.append(fitness)
            metrics["architecture"] = individual
            metrics["generation"] = generation
            metrics["individual"] = individual_index
            metrics["mutation_rate"] = individual.get("mutation_rate", mutation_rate)
            log.append(metrics)
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_individual = individual
                global_best_metrics = metrics
        avg_dice = np.mean([m["dice"] for m in log if m["generation"] == generation])
        avg_flops = np.mean([m["flops"] for m in log if m["generation"] == generation])
        avg_params = np.mean([m["params"] for m in log if m["generation"] == generation])
        norm_flops = avg_flops / 600
        norm_params = avg_params / 10
        performance_gap = 1.0 - avg_dice
        complexity_penalty = (norm_flops + norm_params) / 2.0
        mutation_rate = mutation_rate + 0.2 * complexity_penalty + 0.3 * performance_gap
        mutation_rate = float(np.clip(mutation_rate, 0.05, 0.5))
        num_selections = max(2, population_size // 2)
        selected = select_best_individuals(population, fitness_scores, num_selections)
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover_and_mutate(parent1, parent2, mutation_rate)
            if "mutation_rate" not in child:
                child["mutation_rate"] = mutation_rate
            new_population.append(child)
        population = new_population

    with open("nas_log.json", "w") as f:
        json.dump(log, f, indent=4)

    the_best_model = best_model(
        input_shape=input_shape,
        num_layers=global_best_individual['num_layers'],
        dilation_rate=global_best_individual['dilation_rate'],
        filters=global_best_individual['filters'],
        kernel_size=global_best_individual['kernel_size'],
        activation=global_best_individual['activation'],
        dropout_rate=global_best_individual['dropout_rate'],
        n_classes=n_classes,
        select_conv=global_best_individual['select_conv'],
        select_norm=global_best_individual['select_norm'],
    )
    the_best_model.save("best_model_3D_UNet.h5")

if __name__ == "__main__":
    main()
