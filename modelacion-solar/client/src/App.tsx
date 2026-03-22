import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import PredictionsPage from './pages/PredictionsPage';
import LabPage from './pages/LabPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route path="/" element={<Navigate to="/predictions" replace />} />
          <Route path="/predictions" element={<PredictionsPage />} />
          <Route path="/lab" element={<LabPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
