// ============================================================
// Model Registry — Factory + Strategy Pattern
// ============================================================

import { TrainableModel, ModelState, TrainingMetrics } from './base.model';
import { LinearRegressionModel } from './linear.model';
import { PolynomialRegressionModel } from './polynomial.model';
import { DecisionTreeModel } from './decision-tree.model';
import { SVMModel } from './svm.model';
import { KNNModel } from './knn.model';
import { rSquared, mse, mae, rmse } from '../utils/statistics';

class ModelRegistry {
    private models: Map<string, TrainableModel> = new Map();

    constructor() {
        this.register(new LinearRegressionModel());
        this.register(new PolynomialRegressionModel());
        this.register(new DecisionTreeModel());
        this.register(new SVMModel());
        this.register(new KNNModel());
    }

    register(model: TrainableModel): void {
        this.models.set(model.name, model);
    }

    get(name: string): TrainableModel | undefined {
        return this.models.get(name);
    }

    getAll(): TrainableModel[] {
        return Array.from(this.models.values());
    }

    /**
     * Get summary info for all models (for UI display)
     */
    getCatalog() {
        return this.getAll().map(m => ({
            name: m.name,
            category: m.category,
            description: m.description,
            hyperparameters: m.hyperparameterSchema
        }));
    }

    /**
     * Train a model and compute metrics.
     */
    trainModel(
        modelName: string,
        X: number[][],
        y: number[],
        params: Record<string, number>
    ): { state: ModelState; metrics: TrainingMetrics } {
        const model = this.models.get(modelName);
        if (!model) {
            throw new Error(`Modelo '${modelName}' no encontrado. Modelos disponibles: ${Array.from(this.models.keys()).join(', ')}`);
        }

        const state = model.train(X, y, params);
        const predictions = model.predict(state, X);

        const residuals = y.map((actual, i) => actual - predictions[i]);

        const metrics: TrainingMetrics = {
            r2: rSquared(y, predictions),
            mse: mse(y, predictions),
            mae: mae(y, predictions),
            rmse: rmse(y, predictions),
            predictions,
            residuals
        };

        return { state, metrics };
    }

    /**
     * Compare multiple models on the same data.
     */
    compareModels(
        modelConfigs: { name: string; params: Record<string, number> }[],
        X: number[][],
        y: number[]
    ) {
        return modelConfigs.map(config => {
            try {
                const result = this.trainModel(config.name, X, y, config.params);
                return {
                    name: config.name,
                    params: config.params,
                    metrics: {
                        r2: result.metrics.r2,
                        mse: result.metrics.mse,
                        mae: result.metrics.mae,
                        rmse: result.metrics.rmse
                    },
                    predictions: result.metrics.predictions,
                    error: null
                };
            } catch (err: any) {
                return {
                    name: config.name,
                    params: config.params,
                    metrics: null,
                    predictions: null,
                    error: err.message
                };
            }
        });
    }
}

// Singleton instance
export const modelRegistry = new ModelRegistry();
