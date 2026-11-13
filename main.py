import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Загрузка данных
# -----------------------------
df = pd.read_csv("variant_01.csv")
t = df["index"].to_numpy(dtype=float)
x = df["traffic"].to_numpy(dtype=float)
N = len(x)

print(f"Загружено точек: N = {N}")
print(f"Средний трафик: {np.mean(x):.3f}")
print(f"Мин/макс трафик: {np.min(x):.3f} / {np.max(x):.3f}")

# -----------------------------
# 2. График временного ряда
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(t, x, linewidth=0.8)
plt.xlabel("index")
plt.ylabel("traffic, пакетов/с")
plt.title("Вариант 1: временной ряд сетевого трафика")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("01_traffic_series.png", dpi=150)
plt.show()

# -----------------------------
# 3. Ручное ДПФ для k = 0..50
# -----------------------------
Kmax = 50
a = np.zeros(Kmax + 1)
b = np.zeros(Kmax + 1)
A = np.zeros(Kmax + 1)

# k = 0
a[0] = 2.0 * np.mean(x)          # a0
b[0] = 0.0
A[0] = abs(a[0] / 2.0)           # средний уровень

# k = 1..50
for k in range(1, Kmax + 1):
    cos_part = np.cos(2.0 * np.pi * k * t / N)
    sin_part = np.sin(2.0 * np.pi * k * t / N)

    a[k] = (2.0 / N) * np.sum(x * cos_part)
    b[k] = (2.0 / N) * np.sum(x * sin_part)
    A[k] = np.sqrt(a[k] ** 2 + b[k] ** 2)

# Сохраним коэффициенты в CSV для отчёта
coeff_df = pd.DataFrame({
    "k": np.arange(0, Kmax + 1),
    "a_k": a,
    "b_k": b,
    "A_k": A
})
coeff_df.to_csv("fourier_coefficients_variant_01.csv", index=False)

print("\nКоэффициенты Фурье сохранены в fourier_coefficients_variant_01.csv")
print(f"a0/2 = {a[0]/2:.6f}")

# -----------------------------
# 4. Спектр A_k
# -----------------------------
plt.figure(figsize=(12, 5))
plt.bar(np.arange(1, Kmax + 1), A[1:Kmax + 1], width=0.8)
plt.xlabel("k — номер гармоники")
plt.ylabel("A_k — амплитуда")
plt.title("Амплитудный спектр A_k для k = 1..50")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("02_spectrum.png", dpi=150)
plt.show()

# Топ-10 гармоник по амплитуде (исключая k=0)
top_idx = np.argsort(A[1:])[::-1][:10] + 1
print("\nТоп-10 гармоник по амплитуде A_k:")
print(" k      A_k")
for k in top_idx:
    print(f"{k:2d}  {A[k]:10.4f}")

# Средняя амплитуда и стандартное отклонение по k=1..50
mean_A = np.mean(A[1:])
std_A = np.std(A[1:])
print(f"\nСредняя A_k (k=1..50): {mean_A:.4f}")
print(f"СКО A_k (k=1..50):     {std_A:.4f}")
print(f"Порог mean + 3*std:    {mean_A + 3*std_A:.4f}")

print("\nГармоники, превышающие mean + 3*std:")
for k in range(1, Kmax + 1):
    if A[k] > mean_A + 3 * std_A:
        print(f"  k = {k:2d}, A_k = {A[k]:.4f}")

# -----------------------------
# 5. Тригонометрические приближения K=2 и K=20
# -----------------------------
def trig_poly(K):
    """Возвращает приближение ряда тригонометрическим многочленом порядка K."""
    y = np.full(N, a[0] / 2.0)
    for k in range(1, K + 1):
        y += a[k] * np.cos(2.0 * np.pi * k * t / N)
        y += b[k] * np.sin(2.0 * np.pi * k * t / N)
    return y

y2 = trig_poly(2)
y20 = trig_poly(20)

rmse2 = np.sqrt(np.mean((x - y2) ** 2))
rmse20 = np.sqrt(np.mean((x - y20) ** 2))

print(f"\nRMSE для K=2:  {rmse2:.4f}")
print(f"RMSE для K=20: {rmse20:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(t, x, label="Исходный трафик", linewidth=1.0, alpha=0.9)
plt.plot(t, y2, label=f"K=2 (RMSE={rmse2:.2f})", linewidth=1.5)
plt.plot(t, y20, label=f"K=20 (RMSE={rmse20:.2f})", linewidth=1.5)
plt.xlabel("index")
plt.ylabel("traffic, пакетов/с")
plt.title("Сравнение исходного ряда и тригонометрических приближений")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("03_approximations.png", dpi=150)
plt.show()

# -----------------------------
# 6. Дополнительно: крупный план спектра (k=1..50)
# -----------------------------
plt.figure(figsize=(12, 5))
plt.stem(np.arange(1, Kmax + 1), A[1:Kmax + 1], basefmt=" ")
plt.xlabel("k — номер гармоники")
plt.ylabel("A_k — амплитуда")
plt.title("Спектр A_k (stem-график)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_spectrum_stem.png", dpi=150)
plt.show()

print("\nГотово. Файлы графиков и коэффициентов сохранены в текущей папке.")
