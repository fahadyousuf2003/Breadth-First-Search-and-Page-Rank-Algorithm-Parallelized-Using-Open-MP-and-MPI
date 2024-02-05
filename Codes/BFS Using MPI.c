#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

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
void BFS(int** graph, int numNodes, int startNode, int myRank, int numProcesses);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int myRank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Read graph from file in process 0
    if (myRank == 0) {
        FILE* file = fopen("graph_bfs.txt", "r");
        if (!file) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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

        // Broadcast number of nodes to all processes
        MPI_Bcast(&numNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Scatter graph data to all processes
        for (int i = 0; i < numNodes; i++) {
            MPI_Bcast(graph[i], numNodes, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Specify the start node for BFS
        int startNode;
        printf("\n\n\t\t\t\t\tBreadth First Search Algorithm USING MPI ");
        printf("\n\nEnter the start node for BFS: ");
        scanf("%d", &startNode);
        printf("\n\nStarting Breadth First Search Algorithm From Vertex %d\n\n",startNode);
        // Broadcast start node to all processes
        MPI_Bcast(&startNode, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Execute BFS in all processes
        BFS(graph, numNodes, startNode, myRank, numProcesses);

        // Free allocated memory in process 0
        for (int i = 0; i < numNodes; i++) {
            free(graph[i]);
        }
        free(graph);
    } else {
        // Receive number of nodes from process 0
        int numNodes;
        MPI_Bcast(&numNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate memory for local graph data
        int** graph = (int**)malloc(numNodes * sizeof(int*));
        for (int i = 0; i < numNodes; i++) {
            graph[i] = (int*)malloc(numNodes * sizeof(int));
        }

        // Scatter graph data to all processes
        for (int i = 0; i < numNodes; i++) {
            MPI_Bcast(graph[i], numNodes, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Receive start node from process 0
        int startNode;
        MPI_Bcast(&startNode, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Execute BFS in all processes
        BFS(graph, numNodes, startNode, myRank, numProcesses);

        // Free allocated memory in all processes
        for (int i = 0; i < numNodes; i++) {
            free(graph[i]);
        }
        free(graph);
    }

    MPI_Finalize();

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

void BFS(int** graph, int numNodes, int startNode, int myRank, int numProcesses) {
    bool* visited = (bool*)malloc(numNodes * sizeof(bool));
    for (int i = 0; i < numNodes; i++) {
        visited[i] = false;
    }

    Queue* queue = createQueue();

    visited[startNode] = true;
    enqueue(queue, startNode);

    double startTime = MPI_Wtime();
    double computationTime = 0.0;
    double communicationTime = 0.0;


    // Array to store the visited sequence
    int* visitedSequence = (int*)malloc(numNodes * sizeof(int));
    int visitedIndex = 0;

    while (!isEmpty(queue)) {
        int currentVertex = dequeue(queue);

        // Store the visited vertex in the array
        visitedSequence[visitedIndex++] = currentVertex;

        double computationStartTime = MPI_Wtime();

        for (int i = 0; i < numNodes; i++) {
            if (graph[currentVertex][i] && !visited[i]) {
                visited[i] = true;
                enqueue(queue, i);
            }
        }

        double computationEndTime = MPI_Wtime();
        computationTime += computationEndTime - computationStartTime;

        // Print the contents of the queue
        MPI_Barrier(MPI_COMM_WORLD);
        if (myRank == 0) {
            printf("Queue: ");
            int front = queue->front;
            int rear = queue->rear;
            for (int i = front; i <= rear; i++) {
                printf("%d ", queue->items[i]);
            }
            printf("\n");
        }

        // Print the current vertex
        MPI_Barrier(MPI_COMM_WORLD);
        if (myRank == 0) {
            printf("Visited: ");
            for (int i = 0; i < numNodes; i++) {
                printf("%d ", visited[i]);
            }
            printf("\n");
        }

        // Print the current vertex
        MPI_Barrier(MPI_COMM_WORLD);
        if (myRank == 0) {
            printf("Current Vertex: %d\n\n", currentVertex);
        }
    }

    double endTime = MPI_Wtime();

    // Calculate communication time
    communicationTime = endTime - startTime - computationTime;

    // Print communication and computation times
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == 0) {
        printf("\n\nCommunication Time: %f seconds\n", communicationTime);
        printf("Computation Time: %f seconds\n", computationTime);
    }

    // Print the final sequence of visited vertices
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == 0) {
        printf("\nFinal Visited Sequence: ");
        for (int i = 0; i < numNodes; i++) {
            printf("%d ", visitedSequence[i]);
        }
        printf("\n");

        // Print execution time
        printf("\n\nTotal Time taken: %f seconds\n\n", endTime - startTime);
    }

    // Free allocated memory
    free(visited);
    free(visitedSequence);
    free(queue->items);
    free(queue);
}

