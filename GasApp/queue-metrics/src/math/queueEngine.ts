// Función factorial nativa para evitar dependencias pesadas que rompen Hermes en React Native
const factorial = (n: number): number => {
  if (n < 0) return NaN;
  if (n === 0 || n === 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
};

export type QueueMetrics = {
  rho: number;
  P0: number;
  L: number;
  Lq: number;
  W: number;
  Wq: number;
};

export const calculateMM1 = (lambda: number, mu: number): QueueMetrics => {
  if (lambda >= mu) {
    throw new Error('El sistema es inestable (λ ≥ μ)');
  }

  const rho = lambda / mu;
  const P0 = 1 - rho;
  const L = lambda / (mu - lambda);
  const Lq = Math.pow(rho, 2) / (1 - rho);
  const W = 1 / (mu - lambda);
  const Wq = rho / (mu - lambda);

  return { rho, P0, L, Lq, W, Wq };
};

export const calculateMMS = (lambda: number, mu: number, s: number): QueueMetrics => {
  const rho = lambda / (s * mu);
  if (rho >= 1) {
    throw new Error('El sistema es inestable (ρ ≥ 1)');
  }

  const lambdaMuRatio = lambda / mu;

  // Calcular P0
  let sum = 0;
  for (let n = 0; n < s; n++) {
    sum += Math.pow(lambdaMuRatio, n) / factorial(n);
  }
  const lastTerm = (Math.pow(lambdaMuRatio, s) / factorial(s)) * (1 / (1 - rho));
  const P0 = 1 / (sum + lastTerm);

  // Calcular Lq
  const LqNumerator = Math.pow(lambdaMuRatio, s) * rho;
  const LqDenominator = factorial(s) * Math.pow(1 - rho, 2);
  const Lq = (LqNumerator / LqDenominator) * P0;

  const Wq = Lq / lambda;
  const W = Wq + (1 / mu);
  const L = lambda * W;

  return { rho, P0, L, Lq, W, Wq };
};

export const calculatePn = (lambda: number, mu: number, s: number, n: number, P0: number): number => {
  const lambdaMuRatio = lambda / mu;
  if (s === 1) {
    const rho = lambda / mu;
    return P0 * Math.pow(rho, n);
  } else {
    if (n >= 0 && n <= s) {
      return (Math.pow(lambdaMuRatio, n) / factorial(n)) * P0;
    } else {
      return (Math.pow(lambdaMuRatio, n) / (factorial(s) * Math.pow(s, n - s))) * P0;
    }
  }
};
