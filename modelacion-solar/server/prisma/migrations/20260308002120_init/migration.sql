-- CreateTable
CREATE TABLE "Experiment" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "degree" INTEGER NOT NULL,
    "coefficients" TEXT NOT NULL,
    "r2" REAL,
    "mse" REAL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateTable
CREATE TABLE "DataPoint" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "x" REAL NOT NULL,
    "y" REAL NOT NULL,
    "experimentId" TEXT NOT NULL,
    CONSTRAINT "DataPoint_experimentId_fkey" FOREIGN KEY ("experimentId") REFERENCES "Experiment" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "TrainingResult" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "modelType" TEXT NOT NULL,
    "hyperparams" TEXT NOT NULL,
    "metrics" TEXT NOT NULL,
    "coefficients" TEXT,
    "trainingData" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
