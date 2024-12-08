# Tarea 2
## Descripción
Este programa implementa un algoritmo de ordenamiento que puede ejecutarse en CPU (usando OpenMP) o en GPU (usando CUDA). Dependiendo del modo elegido, el programa utiliza:

MergeSort paralelo (OpenMP) en la CPU.
Quicksort en la GPU utilizando CUDA.

## Requisitos
CUDA: Para ejecutar en GPU.
OpenMP: Para ejecutar en CPU.

## Compilación
Para compilar el programa, basta con ejecutar en una terminal:

```bash
make
```

## Ejecución
El programa se puede ejecutar en dos modos:

Modo CPU (OpenMP)
```
./prog <tamaño_del_vector> <número_de_threads> 0
```
Modo GPU (CUDA)
```
./prog <tamaño_del_vector> <número_de_bloques> 1
```

## Parámetros
- tamaño_del_vector: Tamaño del vector a ordenar.
- número_de_threads/bloques: Número de hilos (CPU) o bloques (GPU) a utilizar.
- modo: 0 para CPU (OpenMP), 1 para GPU (CUDA).

## Ejemplo
```
./prog 1000 4 0   # CPU, con 1000 elementos y 4 hilos
./prog 1000 8 1   # GPU, con 1000 elementos y 8 bloques
```
