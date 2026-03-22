// ============================================================
// Polynomial Regression — Least Squares with configurable degree
// ============================================================

import { TrainableModel, ModelState, HyperparameterDef } from './base.model';
import { gaussElimination } from '../utils/matrix';

export class PolynomialRegressionModel implements TrainableModel {
    readonly name = 'POLYNOMIAL';
    readonly category = 'regression';
    readonly description = 'Regresión Polinomial — Mínimos Cuadrados con grado configurable';
    readonly hyperparameterSchema: HyperparameterDef[] = [
        { name: 'degree', label: 'Grado del Polinomio', type: 'int', min: 1, max: 10, step: 1, default: 2 }
    ];

    train(X: number[][], y: number[], params: Record<string, number>): ModelState {
        const degree = params.degree || 2;
        const n = X.length;
        const m = degree + 1;

        // Use only the first feature column as x
        const xVals = X.map(row => row[0]);

        // Build summation matrix for normal equations
        const sumPowers = new Array(2 * degree + 1).fill(0);
        const sumXY = new Array(m).fill(0);

        for (let i = 0; i < n; i++) {
            for (let p = 0; p <= 2 * degree; p++) {
                sumPowers[p] += Math.pow(xVals[i], p);
            }
            for (let p = 0; p < m; p++) {
                sumXY[p] += y[i] * Math.pow(xVals[i], p);
            }
        }

        const A: number[][] = [];
        for (let i = 0; i < m; i++) {
            const row: number[] = [];
            for (let j = 0; j < m; j++) {
                row.push(sumPowers[i + j]);
            }
            A.push(row);
        }

        const coefficients = gaussElimination(A, sumXY);
        if (!coefficients) {
            throw new Error('Matriz singular. Verifique los datos o reduzca el grado del polinomio.');
        }

        return {
            modelType: this.name,
            coefficients,
            params
        };
    }

    predict(state: ModelState, X: number[][]): number[] {
        const coeffs = state.coefficients!;
        return X.map(row => {
            const x = row[0];
            return coeffs.reduce((sum, c, power) => sum + c * Math.pow(x, power), 0);
        });
    }
}
