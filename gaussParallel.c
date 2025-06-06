#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define TOLERANCE 1e-6
#define N 100
#define THREAD_COUNT 16

double **A, **B;

long usecs(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000000 + t.tv_usec;
}

void display(double **V, int n) {
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            printf("%f ", V[i][j]);
        }
        printf("\n");
    }
}

void initialize(double **M, int n) {
    for (int j = 0; j <= n; j++)
        M[0][j] = 1.0;

    for (int i = 1; i <= n; i++) {
        M[i][0] = 1.0;
        for (int j = 1; j <= n; j++)
            M[i][j] = 0.0;
    }
}

void solve_serial(double **B, int n) {
    printf("\n\n--- Solução em série ---\n\n");
    double diff, tmp;
    int iters = 0;

    for (int iter = 1; iter <= 100; iter++) {
        diff = 0.0;

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                tmp = B[i][j];
                B[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i - 1][j] + B[i][j + 1] + B[i + 1][j]);
                diff += fabs(B[i][j] - tmp);
            }
        }

        iters++;
        printf("u[%d] = %.4f\n", iters, diff);
        if (diff / ((double)N * N) < TOLERANCE) {
            printf("Convergência alcançada na iteração %d\n", iters);
            return;
        }
    }
    printf("\nLimite de iterações alcançado.\n");
}

void solve_parallel(double **A, int n) {
    printf("\n\n--- Solução vermelho e preto ---\n\n");
    double diff, tmp;
    int iters = 0;

    for (int iter = 1; iter <= 100; iter++) {
        diff = 0.0;

        #pragma omp parallel for num_threads(THREAD_COUNT) reduction(+:diff) private(tmp)
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if ((i + j) % 2 == 1) {
                    tmp = A[i][j];
                    A[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i - 1][j] + A[i][j + 1] + A[i + 1][j]);
                    diff += fabs(A[i][j] - tmp);
                }
            }
        }

        #pragma omp parallel for num_threads(THREAD_COUNT) reduction(+:diff) private(tmp)
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if ((i + j) % 2 == 0) {
                    tmp = A[i][j];
                    A[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i - 1][j] + A[i][j + 1] + A[i + 1][j]);
                    diff += fabs(A[i][j] - tmp);
                }
            }
        }

        iters++;
        printf("u[%d] = %.4f\n", iters, diff);
        if (diff / ((double)N * N) < TOLERANCE) {
            printf("Convergência alcançada na iteração %d\n", iters);
            return;
        }
    }

    printf("\nLimite de iterações alcançado.\n");
}

int main() {
    A = (double **)malloc((N + 2) * sizeof(double *));
    B = (double **)malloc((N + 2) * sizeof(double *));
    for (int i = 0; i < N + 2; i++) {
        A[i] = (double *)malloc((N + 2) * sizeof(double));
        B[i] = (double *)malloc((N + 2) * sizeof(double));
    }

    initialize(B, N);
    long t_start = usecs();
    solve_serial(B, N);
    long t_end = usecs();
    printf("%f segundos\n", (t_end - t_start) / 1000000.0);

    initialize(A, N);
    t_start = usecs();
    solve_parallel(A, N - 1);
    t_end = usecs();
    printf("%f segundos\n", (t_end - t_start) / 1000000.0);

    for (int i = 0; i < N + 2; i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);

    return 0;
}