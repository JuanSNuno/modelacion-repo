// ============================================================
// Math Service — Polynomial Regression via Least Squares
// ============================================================

import { gaussElimination } from '../utils/matrix';
import {
    evaluatePolynomial,
    derivePolynomial,
    rSquared,
    mse as calcMSE,
    formatPolynomial
} from '../utils/statistics';

export interface DataPoint {
    x: number;
    y: number;
}

export interface RegressionResult {
    coefficients: number[];
    equation: string;
    r2: number;
    mse: number;
    fittedPoints: DataPoint[];
    derivative?: {
        coefficients: number[];
        equation: string;
        points: DataPoint[];
    };
}

/**
 * Perform polynomial regression using the method of least squares.
 * Builds the normal equations matrix and solves with Gauss elimination.
 *
 * @param data - Array of (x, y) data points
 * @param degree - Degree of the polynomial (1-10)
 * @param computeDerivative - Whether to compute the first derivative
 * @returns RegressionResult with coefficients, metrics, and fitted curve
 */
export function polynomialRegression(
    data: DataPoint[],
    degree: number,
    computeDerivative: boolean = false
): RegressionResult {
    const n = data.length;
    const m = degree + 1; // number of coefficients

    if (n < m) {
        throw new Error(`Se necesitan al menos ${m} puntos para un polinomio de grado ${degree}. Puntos proporcionados: ${n}`);
    }

    // Build the summation matrix for normal equations
    // S[i][j] = Σ(x_k^(i+j)), for i,j = 0..degree
    const sumPowers = new Array(2 * degree + 1).fill(0);
    const sumXY = new Array(m).fill(0);

    for (const { x, y } of data) {
        for (let p = 0; p <= 2 * degree; p++) {
            sumPowers[p] += Math.pow(x, p);
        }
        for (let p = 0; p < m; p++) {
            sumXY[p] += y * Math.pow(x, p);
        }
    }

    // Build the coefficient matrix A and vector b for the normal equations
    const A: number[][] = [];
    for (let i = 0; i < m; i++) {
        const row: number[] = [];
        for (let j = 0; j < m; j++) {
            row.push(sumPowers[i + j]);
        }
        A.push(row);
    }

    // Solve using Gauss Elimination
    const coefficients = gaussElimination(A, sumXY);

    if (!coefficients) {
        throw new Error('La matriz es singular. Revise los datos de entrada (posibles valores duplicados en X).');
    }

    // Compute fitted values at original points
    const actualY = data.map(p => p.y);
    const predictedY = data.map(p => evaluatePolynomial(coefficients, p.x));

    // Compute metrics
    const r2 = rSquared(actualY, predictedY);
    const mseVal = calcMSE(actualY, predictedY);

    // Generate smooth fitted curve
    const xMin = Math.min(...data.map(p => p.x));
    const xMax = Math.max(...data.map(p => p.x));
    const step = (xMax - xMin) / 100;
    const fittedPoints: DataPoint[] = [];
    for (let x = xMin; x <= xMax + step / 2; x += step) {
        fittedPoints.push({ x, y: evaluatePolynomial(coefficients, x) });
    }

    const result: RegressionResult = {
        coefficients,
        equation: formatPolynomial(coefficients),
        r2,
        mse: mseVal,
        fittedPoints
    };

    // Compute derivative if requested (shadow/gnomon mode)
    if (computeDerivative) {
        const derivCoeffs = derivePolynomial(coefficients);
        const derivPoints: DataPoint[] = fittedPoints.map(p => ({
            x: p.x,
            y: evaluatePolynomial(derivCoeffs, p.x)
        }));

        result.derivative = {
            coefficients: derivCoeffs,
            equation: formatPolynomial(derivCoeffs, 'x', 'v'),
            points: derivPoints
        };
    }

    return result;
}
