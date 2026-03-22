// ============================================================
// Model Service — Orchestrates model training and persistence
// ============================================================

import { PrismaClient } from '@prisma/client';
import { modelRegistry } from '../models/registry';

const prisma = new PrismaClient();

export interface TrainRequest {
    modelType: string;
    X: number[][];
    y: number[];
    params: Record<string, number>;
}

export interface CompareRequest {
    models: { name: string; params: Record<string, number> }[];
    X: number[][];
    y: number[];
}

export function trainModel(req: TrainRequest) {
    const { state, metrics } = modelRegistry.trainModel(req.modelType, req.X, req.y, req.params);

    return {
        modelType: req.modelType,
        state,
        metrics: {
            r2: metrics.r2,
            mse: metrics.mse,
            mae: metrics.mae,
            rmse: metrics.rmse
        },
        predictions: metrics.predictions,
        residuals: metrics.residuals
    };
}

export function compareModels(req: CompareRequest) {
    return modelRegistry.compareModels(req.models, req.X, req.y);
}

export function getModelCatalog() {
    return modelRegistry.getCatalog();
}

export async function saveTrainingResult(data: {
    modelType: string;
    hyperparams: Record<string, number>;
    metrics: Record<string, number>;
    coefficients?: any;
    trainingData: { X: number[][]; y: number[] };
}) {
    return prisma.trainingResult.create({
        data: {
            modelType: data.modelType,
            hyperparams: JSON.stringify(data.hyperparams),
            metrics: JSON.stringify(data.metrics),
            coefficients: data.coefficients ? JSON.stringify(data.coefficients) : null,
            trainingData: JSON.stringify(data.trainingData)
        }
    });
}

export async function getTrainingHistory() {
    const results = await prisma.trainingResult.findMany({
        orderBy: { createdAt: 'desc' },
        take: 50
    });
    return results.map(r => ({
        ...r,
        hyperparams: JSON.parse(r.hyperparams),
        metrics: JSON.parse(r.metrics),
        coefficients: r.coefficients ? JSON.parse(r.coefficients) : null
    }));
}
