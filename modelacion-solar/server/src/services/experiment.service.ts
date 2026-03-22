// ============================================================
// Experiment Service — CRUD operations via Prisma
// ============================================================

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export interface CreateExperimentDTO {
    name: string;
    type: 'LUZ' | 'SOMBRA';
    degree: number;
    coefficients: number[];
    r2: number | null;
    mse: number | null;
    dataPoints: { x: number; y: number }[];
}

export async function createExperiment(data: CreateExperimentDTO) {
    return prisma.experiment.create({
        data: {
            name: data.name,
            type: data.type,
            degree: data.degree,
            coefficients: JSON.stringify(data.coefficients),
            r2: data.r2,
            mse: data.mse,
            dataPoints: {
                create: data.dataPoints.map(dp => ({ x: dp.x, y: dp.y }))
            }
        },
        include: { dataPoints: true }
    });
}

export async function getAllExperiments() {
    return prisma.experiment.findMany({
        orderBy: { createdAt: 'desc' },
        include: { dataPoints: true }
    });
}

export async function getExperimentById(id: string) {
    return prisma.experiment.findUnique({
        where: { id },
        include: { dataPoints: true }
    });
}

export async function deleteExperiment(id: string) {
    return prisma.experiment.delete({ where: { id } });
}
