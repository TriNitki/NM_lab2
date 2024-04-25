from utils import Input
import json

from methods import (ort, solve, inv, solve_iter, get_nev_vector, 
                     norm_nev_vector, get_nev_matrix, norm_nev_matrix)

def main():
    with open("input.json", "r") as f:
        user_data = json.load(f)
    
    inp = Input(
        user_data["task_type"], 
        user_data["matrix"], 
        user_data["vector"]
    )
    
    orted = ort(inp)
    slau = solve(orted, inp.vector)
    iter = solve_iter(inp)
    inver = inv(orted)
    nev = get_nev_vector(inp, slau)
    norm_vector = norm_nev_vector(nev)
    norm_mat = get_nev_matrix(inp)
    normed_mat_nev = norm_nev_matrix(norm_mat.tolist())

    print("Ортагонализированная матрица:\n", orted, end="\n\n")
    print("Решение по СЛАУ:\n", slau, end="\n\n")
    print("Решение по методом простой итерации:\n", iter, end="\n\n")
    print("Инвертированная матрица:\n", inver, end="\n\n")
    print("Вектор невязки:\n", nev, end="\n\n")
    print("Норма вектора невязки:\n", norm_vector, end="\n\n")
    print("Матрица невязки:\n", norm_mat, end="\n\n")
    print("Норма матрицы невязки:\n", normed_mat_nev, end="\n\n")
    
    
    
if __name__ == "__main__":
    main()