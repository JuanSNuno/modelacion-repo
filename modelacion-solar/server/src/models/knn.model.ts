// ============================================================
// K-Nearest Neighbors Regressor
// ============================================================

import { TrainableModel, ModelState, HyperparameterDef } from './base.model';
import { euclideanDistance } from '../utils/matrix';

export class KNNModel implements TrainableModel {
    readonly name = 'KNN';
    readonly category = 'regression';
    readonly description = 'K-Nearest Neighbors — Regresión por vecinos más cercanos';
    readonly hyperparameterSchema: HyperparameterDef[] = [
        { name: 'k', label: 'K (Número de Vecinos)', type: 'int', min: 1, max: 50, step: 1, default: 3 },
        { name: 'weighted', label: 'Ponderado por Distancia', type: 'select', default: 1, options: [{ value: 0, label: 'Uniforme' }, { value: 1, label: 'Por Distancia' }] }
    ];

    train(X: number[][], y: number[], params: Record<string, number>): ModelState {
        // KNN is a lazy learner — just store the training data
        return {
            modelType: this.name,
            trainingData: { X, y },
            params
        };
    }

    predict(state: ModelState, X: number[][]): number[] {
        const { X: trainX, y: trainY } = state.trainingData!;
        const k = Math.min(state.params.k || 3, trainX.length);
        const weighted = state.params.weighted === 1;

        return X.map(point => {
            // Compute distances to all training points
            const distances = trainX.map((trainPoint, i) => ({
                distance: euclideanDistance(point, trainPoint),
                y: trainY[i]
            }));

            // Sort by distance and take k nearest
            distances.sort((a, b) => a.distance - b.distance);
            const neighbors = distances.slice(0, k);

            if (weighted) {
                // Inverse distance weighting
                const totalWeight = neighbors.reduce((s, n) => {
                    return s + (n.distance < 1e-10 ? 1e10 : 1 / n.distance);
                }, 0);
                return neighbors.reduce((s, n) => {
                    const w = n.distance < 1e-10 ? 1e10 : 1 / n.distance;
                    return s + (w / totalWeight) * n.y;
                }, 0);
            }

            // Uniform: simple average
            return neighbors.reduce((s, n) => s + n.y, 0) / k;
        });
    }
}
