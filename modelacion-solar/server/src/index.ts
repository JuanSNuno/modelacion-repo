// ============================================================
// Express Entry Point
// ============================================================

import express from 'express';
import cors from 'cors';
import modelRoutes from './routes/model.routes';
import experimentRoutes from './routes/experiment.routes';
import labRoutes from './routes/lab.routes';
import { errorHandler } from './middleware/error-handler';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Routes
app.use('/api/model', modelRoutes);
app.use('/api/experiments', experimentRoutes);
app.use('/api/lab', labRoutes);

// Health check
app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handler
app.use(errorHandler);

app.listen(PORT, () => {
    console.log(`🌞 SolarMotion API running on http://localhost:${PORT}`);
});

export default app;
