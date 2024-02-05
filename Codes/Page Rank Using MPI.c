#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

// Function to perform matrix-vector multiplication: new_pr = A * pr
void matrix_vector_multiply(int n, double* pr, double* new_pr, int* adj_matrix) {
    for (int i = 0; i < n; i++) {
        new_pr[i] = 0.0;
        for (int j = 0; j < n; j++) {
            new_pr[i] += adj_matrix[i * n + j] * pr[j];
        }
    }
}

void pagerank(int n, double* pr, int* adj_matrix, int num_processes, int max_iterations, const char* output_filename, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    double* new_pr = (double*)malloc(n * sizeof(double));
    double* global_vector = (double*)malloc(n * sizeof(double));
    int converged;

    FILE* output_file = NULL;
    if (output_filename != NULL) {
        output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Error opening output file");
            exit(EXIT_FAILURE);
        }
    }

    double total_computation_time = 0.0;
    double total_communication_time = 0.0;
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < max_iterations; iter++) {
        double computation_start_time = MPI_Wtime();

        converged = 1;

        // Perform local PageRank calculations using matrix-vector multiplication
        matrix_vector_multiply(n, pr, new_pr, adj_matrix);

        
        for (int i = 0; i < n; i++) {
            new_pr[i] = (1 - DAMPING_FACTOR) / n + DAMPING_FACTOR * new_pr[i];
        }

        double computation_end_time = MPI_Wtime();
        double computation_time = computation_end_time - computation_start_time;

        // Exchange information between processes
        MPI_Barrier(comm); // Synchronize processes before communication
        double communication_start = MPI_Wtime();
        MPI_Allreduce(new_pr, global_vector, n, MPI_DOUBLE, MPI_SUM, comm);
        double communication_end = MPI_Wtime();
        double communication_time = communication_end - communication_start;

        total_computation_time += computation_time;
        total_communication_time += communication_time;

        // Check for convergence
        
        for (int i = 0; i < n; i++) {
            if (fabs(global_vector[i] - pr[i]) > EPSILON) {
                converged = 0;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_LAND, comm);

        // Update PageRank scores
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            pr[i] = global_vector[i];
        }

        // Normalize PageRank scores after each iteration
        double norm_factor = 0.0;
        
        for (int i = 0; i < n; i++) {
            norm_factor += pr[i];
        }
        norm_factor = 1.0 / norm_factor;

 
        for (int i = 0; i < n; i++) {
            pr[i] *= norm_factor;
        }

        double end_time = MPI_Wtime();
        double runtime = end_time - start_time;

        if (converged || iter == max_iterations - 1) {
            if (rank == 0) {
                printf("\nFinal PageRank Scores (Iteration %d):\n", iter + 1);
                for (int i = 0; i < n; i++) {
                    printf("Node %-3d: %.4f\n", i, pr[i]);
                }
                printf("\nTotal Runtime for the Final Iteration: %.4f seconds\n", runtime);
                printf("Total Computation Time: %.4f seconds\n", total_computation_time);
                printf("Total Communication Time: %.4f seconds\n", total_communication_time);

                if (output_file != NULL) {
                    fprintf(output_file, "\nFinal PageRank Scores (Iteration %d):\n", iter + 1);
                    for (int i = 0; i < n; i++) {
                        fprintf(output_file, "Node %-3d: %.4f\n", i, pr[i]);
                    }
                    fprintf(output_file, "\nTotal Runtime for the Final Iteration: %.4f seconds\n", runtime);
                    fprintf(output_file, "Total Computation Time: %.4f seconds\n", total_computation_time);
                    fprintf(output_file, "Total Communication Time: %.4f seconds\n", total_communication_time);
                }
            }

            if (converged) {
                break; // Exit the loop when converged
            }
        }
    }

    if (output_file != NULL) {
        fclose(output_file);
    }

    free(new_pr);
    free(global_vector);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int num_processes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file> <output_file> <max_vertices>\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int max_vertices = atoi(argv[3]);

    if (max_vertices > MAX_VERTICES) {
        if (rank == 0) {
            fprintf(stderr, "Error: max_vertices exceeds the maximum allowed value (%d).\n", MAX_VERTICES);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int n;
    int adj_matrix[MAX_VERTICES][MAX_VERTICES] = {0};

    if (rank == 0) {
        read_graph_from_file(input_filename, &n, (int*)adj_matrix);
        if (n > max_vertices) {
            fprintf(stderr, "Error: The graph size exceeds the specified max_vertices.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(adj_matrix, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    double pr[MAX_VERTICES];

    // Initialize PageRank scores
    for (int i = 0; i < n; i++) {
        pr[i] = 1.0 / n;
    }

    double start_time = MPI_Wtime();

    pagerank(n, pr, (int*)adj_matrix, num_processes, MAX_ITERATIONS, output_filename, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double total_runtime = end_time - start_time;

    if (rank == 0) {
        printf("\nTotal Runtime: %.4f seconds\n", total_runtime);

        // Display the graph (up to max_vertices)
        display_graph(n, (int*)adj_matrix);
    }

    MPI_Finalize();

    return 0;
}

