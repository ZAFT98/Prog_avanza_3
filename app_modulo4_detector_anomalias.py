import time
import numpy as np


def generar_datos(n, seed=42):
    rng = np.random.default_rng(seed)

    temperaturas = rng.uniform(15, 40, n)
    humedades = rng.uniform(20, 80, n)

    return temperaturas, humedades


def generar_datos_con_dia(n, seed=42):
    rng = np.random.default_rng(seed)

    temperaturas, humedades = generar_datos(n, seed)

    # 0 = lunes ... 6 = domingo
    dias = rng.integers(0, 7, n)

    es_fin_de_semana = (dias == 5) | (dias == 6)

    return temperaturas, humedades, es_fin_de_semana


# Loop
def alarma_con_loop_v2(temperaturas, humedades, es_fin_de_semana):
    resultados = []
    operaciones = 0

    for temp, hum, fin_semana in zip(
        temperaturas, humedades, es_fin_de_semana
    ):
        operaciones += 1

        P = temp > 30
        Q = hum < 40
        R = fin_semana

        resultados.append(P and Q and (not R))

    return np.array(resultados), operaciones


# NumPy
def alarma_vectorizada_v2(temperaturas, humedades, es_fin_de_semana):
    P = temperaturas > 30
    Q = humedades < 40
    R = es_fin_de_semana

    return P & Q & (~R)


# --------------------------------------------------
# Verificación
# --------------------------------------------------

n = 10_000

temps, hums, finde = generar_datos_con_dia(n)

resultados_loop, operaciones = alarma_con_loop_v2(
    temps, hums, finde
)

resultados_vec = alarma_vectorizada_v2(
    temps, hums, finde
)

coinciden = np.array_equal(
    resultados_loop,
    resultados_vec
)

print(f"¿Coinciden loop y vectorizado?: {coinciden}")


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

n = 1_000_000

temps_n, hums_n, finde_n = generar_datos_con_dia(n)


inicio = time.perf_counter()

resultados_loop, operaciones = alarma_con_loop_v2(
    temps_n, hums_n, finde_n
)

t_loop = time.perf_counter() - inicio


inicio = time.perf_counter()

resultados_vec = alarma_vectorizada_v2(
    temps_n, hums_n, finde_n
)

t_vec = time.perf_counter() - inicio


aceleracion = t_loop / t_vec


print("\n--- Benchmark ---")
print(f"n = {n:,}")
print(f"Operaciones lógicas = {operaciones:,}")
print(f"Tiempo loop = {t_loop:.5f} s")
print(f"Tiempo NumPy = {t_vec:.5f} s")
print(f"NumPy es aproximadamente {aceleracion:.1f}x más rápido")
