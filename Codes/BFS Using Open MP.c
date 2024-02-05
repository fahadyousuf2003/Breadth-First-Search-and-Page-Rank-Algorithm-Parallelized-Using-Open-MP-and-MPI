#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_NODES 1000

typedef struct {
    int front, rear;
    int* items;
} Queue;

// Function declarations
Queue* createQueue();
void enqueue(Queue* queue, int value);
int dequeue(Queue* queue);
bool isEmpty(Queue* queue);
void BFS(int** graph, int numNodes, int startNode);

int main() {
    // Read graph from file
    FILE* file = fopen("graph_bfs.txt", "r");
    if (!file) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    int numNodes;
    fscanf(file, "%d", &numNodes);

    int** graph = (int**)malloc(numNodes * sizeof(int*));
    for (int i = 0; i < numNodes; i++) {
        graph[i] = (int*)malloc(numNodes * sizeof(int));
        for (int j = 0; j < numNodes; j++) {
            fscanf(file, "%d", &graph[i][j]);
        }
    }

    fclose(file);

    // Specify the start node for BFS
    int startNode;
    printf("\n\n\t\t\t\t\tBreadth First Search Algorithm USING OPEN-MP ");
    printf("\n\nEnter the start node for BFS: ");
    scanf("%d", &startNode);

    // Execute BFS
    BFS(graph, numNodes, startNode);

    // Free allocated memory
    for (int i = 0; i < numNodes; i++) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}

Queue* createQueue() {
    Queue* queue = (Queue*)malloc(sizeof(Queue));
    queue->front = -1;
    queue->rear = -1;
    queue->items = (int*)malloc(MAX_NODES * sizeof(int));
    return queue;
}

void enqueue(Queue* queue, int value) {
    if (queue->rear == MAX_NODES - 1) {
        printf("Queue is full.\n");
        return;
    }

    if (queue->front == -1) {
        queue->front = 0;
    }

    queue->rear++;
    queue->items[queue->rear] = value;
}

int dequeue(Queue* queue) {
    if (queue->front == -1) {
        printf("Queue is empty.\n");
        return -1;
    }

    int item = queue->items[queue->front];
    queue->front++;

    if (queue->front > queue->rear) {
        queue->front = queue->rear = -1;
    }

    return item;
}

bool isEmpty(Queue* queue) {
    return queue->front == -1;
}

void BFS(int** graph, int numNodes, int startNode) {
    bool* visited = (bool*)malloc(numNodes * sizeof(bool));
    for (int i = 0; i < numNodes; i++) {
        visited[i] = false;
    }

    Queue* queue = createQueue();

    visited[startNode] = true;
    enqueue(queue, startNode);

    double startTime = omp_get_wtime();
    double computationTime = 0.0;
    double communicationTime = 0.0;

    printf("\nBreadth-First Search starting from node %d:\n\n", startNode);

    // Array to store the visited sequence
    int* visitedSequence = (int*)malloc(numNodes * sizeof(int));
    int visitedIndex = 0;

    while (!isEmpty(queue)) {
        int currentVertex = dequeue(queue);

        // Store the visited vertex in the array
        visitedSequence[visitedIndex++] = currentVertex;

        double computationStartTime = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            if (graph[currentVertex][i] && !visited[i]) {
                #pragma omp critical
                {
                    visited[i] = true;
                    enqueue(queue, i);
                }
            }
        }

        double computationEndTime = omp_get_wtime();
        computationTime += computationEndTime - computationStartTime;

        // Print the contents of the queue
        #pragma omp barrier
        #pragma omp single
        {
            printf("Queue: ");
            int front = queue->front;
            int rear = queue->rear;
            for (int i = front; i <= rear; i++) {
                printf("%d ", queue->items[i]);
            }
            printf("\n");
        }

        #pragma omp barrier
        #pragma omp single
        {
            printf("Visited: ");
            for (int i = 0; i < numNodes; i++) {
                printf("%d ", visited[i]);
            }
            printf("\n");
        }

        // Print the current vertex
        #pragma omp barrier
        #pragma omp single
        {
            printf("Current Vertex: %d\n\n", currentVertex);
        }
    }

    double endTime = omp_get_wtime();

    // Estimate communication time based on critical section
    communicationTime = endTime - startTime - computationTime;

    // Print communication and computation times
    printf("\nCommunication Time: %f seconds\n", communicationTime);
    printf("Computation Time: %f seconds\n", computationTime);

    // Print the final sequence of visited vertices
    printf("\nFinal Visited Sequence: ");
    for (int i = 0; i < numNodes; i++) {
        printf("%d ", visitedSequence[i]);
    }
    printf("\n");

    // Print execution time
    printf("\nTotal Time taken: %f seconds\n", endTime - startTime);
    printf("\n");

    // Free allocated memory
    free(visited);
    free(visitedSequence);
    free(queue->items);
    free(queue);
}

