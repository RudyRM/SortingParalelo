#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>
using namespace std;


__global__ void fase1_particion_kernel(int* d_arreglo, int* d_aux, int tamano, int fase, int indice_pivote) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tamano) return;

    // Memoria compartida para almacenar el pivote
    __shared__ int pivote;
    if (threadIdx.x == 0) {
        if (indice_pivote == 0) { pivote = d_arreglo[0]; }
        else { pivote = d_arreglo[indice_pivote + 1]; }
    }
    __syncthreads();

    // Rearreglar los elementos
    __shared__ int izquierda_temp[1024];
    __shared__ int derecha_temp[1024];
    __shared__ int cuenta_izquierda, cuenta_derecha;

    if (threadIdx.x == 0) {
        cuenta_izquierda = 0;
        cuenta_derecha = 0;
    }
    __syncthreads();

	// Contar mayores y menores al pivote
    if (tid >= d_aux[fase] && tid < tamano) {
        if (d_arreglo[tid] < pivote) {
            int pos = atomicAdd(&cuenta_izquierda, 1);
            izquierda_temp[pos] = d_arreglo[tid];
        } else {
            int pos = atomicAdd(&cuenta_derecha, 1);
            derecha_temp[pos] = d_arreglo[tid];
        }
    }
    __syncthreads();

    // Escribir los resultados de vuelta en la memoria global
    if (tid < cuenta_izquierda) {
        d_arreglo[d_aux[fase] + tid] = izquierda_temp[tid];
    } else if (tid < cuenta_izquierda + cuenta_derecha) {
        d_arreglo[d_aux[fase] + tid] = derecha_temp[tid - cuenta_izquierda];
    }

    if (threadIdx.x == 0) {
        d_aux[fase + 1] = d_aux[fase] + cuenta_izquierda;
    }
}


__device__ void quicksort_iterativo(int *arreglo, int izquierda, int derecha) {
    // Crear una pila explícita para manejar los límites
    int pila[1024]; // Asegúrate de que sea lo suficientemente grande
    int cima = -1;

    // Push inicial para los límites
    pila[++cima] = izquierda;
    pila[++cima] = derecha;

    while (cima >= 0) {
        // Pop de los límites
        derecha = pila[cima--];
        izquierda = pila[cima--];

        int pivote = arreglo[izquierda + (derecha - izquierda) / 2];
        int i = izquierda;
        int j = derecha;

        // Partición
        while (i <= j) {
            while (arreglo[i] < pivote) i++;
            while (arreglo[j] > pivote) j--;

            if (i <= j) {
                int temp = arreglo[i];
                arreglo[i] = arreglo[j];
                arreglo[j] = temp;
                i++;
                j--;
            }
        }

        // Push de los subarreglos a la pila
        if (izquierda < j) {
            pila[++cima] = izquierda;
            pila[++cima] = j;
        }
        if (i < derecha) {
            pila[++cima] = i;
            pila[++cima] = derecha;
        }
    }
}

__global__ void fase2_quicksort(int *A, int *posiciones_pivote, int num_intervalos) {
    int indice_bloque = blockIdx.x;
    if (indice_bloque >= num_intervalos) return;

    int inicio = posiciones_pivote[indice_bloque];
    int fin = posiciones_pivote[indice_bloque + 1] - 1;

    quicksort_iterativo(A, inicio, fin); // Llama a la versión iterativa
}

// Función para medir el tiempo en CUDA
double cudaTimer(int* d_arreglo, int* d_aux, vector<int>& h_arreglo, vector<int>& h_aux, int tamano, int nb, int hilos_por_bloque) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int fase = 0; fase < nb - 1; ++fase) {
        int indice_pivote = h_aux[fase];
        fase1_particion_kernel<<<nb, hilos_por_bloque>>>(d_arreglo, d_aux, tamano, fase, indice_pivote);
        cudaDeviceSynchronize();
    }

    int num_intervalos = h_aux.size() - 1; // Número de intervalos
    fase2_quicksort<<<num_intervalos, 1>>>(d_arreglo, d_aux, num_intervalos);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0; // Retorna el tiempo en segundos
}

// Implementación de mergeSort paralela
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Función recursiva para Merge Sort
void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp task shared(arr) if (right - left > 1000)
        mergeSort(arr, left, mid);

        #pragma omp task shared(arr) if (right - left > 1000)
        mergeSort(arr, mid + 1, right);

        #pragma omp taskwait
        merge(arr, left, mid, right);
    }
}


// Generar un vector aleatorio
vector<int> generateRandomVector(int n) {
    vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = rand() % 10000; // Números aleatorios entre 0 y 9999
    }
    return v;
}

void imprimir_arreglo(const vector<int>& arreglo, const string& etiqueta) {
    cout << etiqueta << ": ";
    for (int val : arreglo) {
        cout << val << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Uso: " << argv[0] << " <tamaño del vector inicial> <número de threads/bloques> <modo (0 CPU, 1 GPU)>" << endl;
        return 1;
    }
    
    int modo = atoi(argv[3]);
	int n = atoi(argv[1]);   // Tamaño inicial del vector
	
	if (modo == 0) {
		int nt = atoi(argv[2]);  // Número de threads

		if (n <= 0 || nt <= 0) {
		    cerr << "El tamaño del vector y el número de threads deben ser mayores a 0." << endl;
		    return 1;
		}

		// Configurar número de threads en OpenMP
		omp_set_num_threads(nt);

		srand(time(0)); // Semilla para números aleatorios

		double promedio_parallel = 0, promedio_std = 0;

	    cout << "Tamaño del vector: " << n << endl;
	    vector<int> arr = generateRandomVector(n);
	    vector<int> arr_copy = arr;

	    // Medir tiempos para mergeSort paralelo
	    for (int i = 0; i < 5; i++) {
	        vector<int> temp = arr; // Copia para cada ejecución
	        double t = omp_get_wtime();
	        #pragma omp parallel
	        {
	            #pragma omp single
	            mergeSort(temp, 0, n - 1);
	        }
	        promedio_parallel += omp_get_wtime() - t;
	    }

	    // Medir tiempos para std::sort
	    for (int i = 0; i < 5; i++) {
	        vector<int> temp = arr_copy; // Copia para cada ejecución
	        double t = omp_get_wtime();
	        sort(temp.begin(), temp.end());
	        promedio_std += omp_get_wtime() - t;
	    }

	    promedio_parallel /= 5;
	    promedio_std /= 5;
	    
	    // Calcular y mostrar promedios
	    cout << "Promedio paralelo: " << promedio_parallel << " segundos" << endl;
	    cout << "Promedio std::sort: " << promedio_std << " segundos" << endl;
		
    	return 0;
    }
    
    if(modo == 1){
		int nb = atoi(argv[2]); // Número de bloques

		int hilos_por_bloque = 1024;
		if (nb == 0) { nb = (n + hilos_por_bloque - 1) / hilos_por_bloque; }
		srand(time(0)); // Inicializar semilla de random

	    cout << "Tamaño del vector: " << n << endl;

	    double promedio_cuda = 0, promedio_std = 0;

	    for (int iter = 0; iter < 5; ++iter) {
	        vector<int> h_arreglo = generateRandomVector(n);
	        vector<int> h_aux(nb + 1, 0);
	        h_aux[nb] = n;

	        // Memoria en el dispositivo
	        int* d_arreglo;
	        int* d_aux;
	        cudaMalloc(&d_arreglo, n * sizeof(int));
	        cudaMalloc(&d_aux, h_aux.size() * sizeof(int));

	        cudaMemcpy(d_arreglo, h_arreglo.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	        cudaMemcpy(d_aux, h_aux.data(), h_aux.size() * sizeof(int), cudaMemcpyHostToDevice);

	        // Medir tiempo para CUDA
	        promedio_cuda += cudaTimer(d_arreglo, d_aux, h_arreglo, h_aux, n, nb, hilos_por_bloque);

	        // Liberar memoria en dispositivo
	        cudaFree(d_arreglo);
	        cudaFree(d_aux);

	        // Medir tiempo para std::sort
	        vector<int> copy = h_arreglo;
	        double start = clock();
	        sort(copy.begin(), copy.end());
	        double end = clock();
	        promedio_std += (end - start) / CLOCKS_PER_SEC;
	    }

	    // Calcular promedios
	    promedio_cuda /= 5;
	    promedio_std /= 5;

	    cout << "Promedio CUDA: " << promedio_cuda << " segundos" << endl;
	    cout << "Promedio std::sort: " << promedio_std << " segundos" << endl;
		
    	return 0;
	}

    return 0;
}
