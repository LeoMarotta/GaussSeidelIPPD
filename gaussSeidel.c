#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100             // Número de pontos na barra
#define MAX_IT 1000       // Número máximo de iterações
#define TOL 1e-6          // Tolerância para critério de parada

void gauss_seidel(double* u, int n, int max_it, double tol) {
    for (int it = 0; it < max_it; it++) {
        double max_diff = 0.0;

        for (int i = 1; i < n - 1; i++) {
            double old = u[i];
            u[i] = 0.5 * (u[i - 1] + u[i + 1]);
            double diff = fabs(u[i] - old);
            if (diff > max_diff)
                max_diff = diff;
        }

        if (max_diff < tol) {
            printf("Convergência alcançada na iteração %d\n", it);
            break;
        }
    }
}

int main() {
    double* u = malloc(N * sizeof(double));

    // Condições iniciais e de contorno
    for (int i = 0; i < N; i++) {
        u[i] = 0.0;
    }
    u[0] = 100.0;      // Extremidade esquerda
    u[N - 1] = 50.0;   // Extremidade direita

    gauss_seidel(u, N, MAX_IT, TOL);

    // Imprime a solução final
    for (int i = 0; i < N; i++) {
        printf("u[%d] = %.4f\n", i, u[i]);
    }

    free(u);
    return 0;
}

