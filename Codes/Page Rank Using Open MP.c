#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define MAX_VERTICES 100
#define DAMPING_FACTOR 0.85
#define EPSILON 1e-6
#define MAX_ITERATIONS 100

void generate_random_graph(int n, int* adj_matrix) {
    srand(time(NULL));

    // Generate a random directed graph (sparse)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && rand() % 4 == 0) {
                adj_matrix[i * n + j] = 1;
            }
        }
    }
}

void read_graph_from_file(const char* filename, int* n, int* adj_matrix) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", n);

    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            fscanf(file, "%d", &adj_matrix[i * (*n) + j]);
        }
    }

    fclose(file);
}

void display_graph(int n, int* adj_matrix) {
    printf("Graph Adjacency Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", adj_matrix[i * n + j]);
        }
        printf("\n");
    }
}

void pagerank(int n, double* pr, int* adj_matrix, int num_threads, int max_iterations, const char* output_filename) {
    double* new_pr = (double*)malloc(n * sizeof(double));
    int converged;

    double total_computation_time = 0.0;
    double total_synchronization_time = 0.0;
    double total_communication_time = 0.0;

    FILE* output_file = NULL;
    if (output_filename != NULL) {
        output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Error opening output file");
            exit(EXIT_FAILURE);
        }
    }

    printf("%-6s %-12s %-12s %-12s %-8s\n", "Iter", "Runtime", "Computation", "Communication", "Converged");
    printf("------------------------------------------------------------\n");

    for (int iter = 0; iter < max_iterations; iter++) {
        double start_time = omp_get_wtime();

        converged = 1;

        double local_computation_time = 0.0;
        double local_communication_time = 0.0;

        #pragma omp parallel num_threads(num_threads) shared(new_pr) reduction(&&:converged) reduction(+:local_computation_time) reduction(+:local_communication_time)
        {
            double local_start_time = omp_get_wtime();

            #pragma omp for
            for (int i = 0; i < n; i++) {
                double contribution = (1 - DAMPING_FACTOR) / n;

                for (int j = 0; j < n; j++) {
                    if (adj_matrix[j * n + i] != 0) {
                        contribution += DAMPING_FACTOR * pr[j] / (unsigned int)__builtin_popcount((unsigned int)(adj_matrix[j * n + i]));
                    }
                }

                new_pr[i] = contribution;

                // Measure local computation time
                local_computation_time += omp_get_wtime() - local_start_time;
            }

            // Measure local communication time (copying new_pr to pr)
            double local_comm_start_time = omp_get_wtime();

            #pragma omp for
            for (int i = 0; i < n; i++) {
                pr[i] = new_pr[i];
            }

            double local_comm_end_time = omp_get_wtime();
            local_communication_time += local_comm_end_time - local_comm_start_time;

            // Ensure all threads have finished their work before proceeding
            #pragma omp barrier

            // Measure synchronization time (barrier)
            double sync_start_time = omp_get_wtime();

            double sync_end_time = omp_get_wtime();
            #pragma omp atomic update
            total_synchronization_time += sync_end_time - sync_start_time;

            // Use a global variable to control termination across all threads
            #pragma omp single
            {
                if (converged) {
                    iter = max_iterations; // Exit the loop when converged
                }
            }
        }

        double end_time = omp_get_wtime();
        double runtime = end_time - start_time;

        // Measure total computation time (excluding synchronization)
        total_computation_time += local_computation_time;

        // Measure total communication time
        total_communication_time += local_communication_time;

        printf("%-6d %-12.4f %-12.4f %-12.4f %-8d\n", iter + 1, runtime, total_computation_time, total_communication_time, converged);

        if (output_file != NULL) {
            fprintf(output_file, "Iteration %d:\n", iter + 1);
            for (int i = 0; i < n; i++) {
                fprintf(output_file, "Node %-3d: %.4f\n", i, pr[i]);
            }
            fprintf(output_file, "\n");
        }

        if (converged) {
            break; // Exit the loop when converged
        }
    }

    if (output_file != NULL) {
        fclose(output_file);
    }

    free(new_pr);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads> <max_vertices> <output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* input_filename = argv[1];
    int num_threads = atoi(argv[2]);
    int max_vertices = atoi(argv[3]);
    const char* output_filename = argv[4];

    if (max_vertices > MAX_VERTICES) {
        fprintf(stderr, "Error: max_vertices exceeds the maximum allowed value (%d).\n", MAX_VERTICES);
        exit(EXIT_FAILURE);
    }

    int n;
    int adj_matrix[MAX_VERTICES][MAX_VERTICES] = {0};

    read_graph_from_file(input_filename, &n, (int*)adj_matrix);

    if (n > max_vertices) {
        fprintf(stderr, "Error: The graph size exceeds the specified max_vertices.\n");
        exit(EXIT_FAILURE);
    }

    double pr[MAX_VERTICES];

    // Initialize PageRank scores
    for (int i = 0; i < n; i++) {
        pr[i] = 1.0 / n;
    }

    double total_synchronization_time = 0.0;

    double start_time = omp_get_wtime();

    pagerank(n, pr, (int*)adj_matrix, num_threads, MAX_ITERATIONS, output_filename);

    double end_time = omp_get_wtime();
    double total_runtime = end_time - start_time;

    printf("\nFinal PageRank Scores:\n");
    for (int i = 0; i < n; i++) {
        printf("Node %-3d: %.4f\n", i, pr[i]);
    }

    printf("\nTotal Runtime: %.4f seconds\n", total_runtime);
    printf("Total Computation Time: %.4f seconds\n", total_runtime - total_synchronization_time);
    printf("Total Synchronization Time: %.4f seconds\n", total_synchronization_time);

    // Display the graph (up to max_vertices)
    display_graph(n, (int*)adj_matrix);

    return 0;
}

