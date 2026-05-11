import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Text, useTheme, Surface, Card, Button, TextInput } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { calculatePn } from '../math/queueEngine';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Results'>;
  route: RouteProp<RootStackParamList, 'Results'>;
};

export default function ResultsScreen({ navigation, route }: Props) {
  const theme = useTheme();
  const { studyId } = route.params;

  const [model, setModel] = useState<any>(null);
  const [study, setStudy] = useState<any>(null);
  const [nInput, setNInput] = useState('0');
  const [pnResult, setPnResult] = useState<number | null>(null);
  const [recommendation, setRecommendation] = useState('');
  const [customNotes, setCustomNotes] = useState('');

  useEffect(() => {
    const studyData = db.getFirstSync('SELECT * FROM studies WHERE id = ?', [studyId]);
    const modelData = db.getFirstSync('SELECT * FROM queue_models WHERE study_id = ?', [studyId]);
    setStudy(studyData);
    setModel(modelData);

    if (modelData) {
      if ((modelData as any).result_Rho > 0.85) {
        setRecommendation('⚠️ La utilización (ρ) es muy alta (>85%). Se recomienda agregar un servidor o reducir los tiempos de atención para evitar cuellos de botella.');
      } else if ((modelData as any).result_Rho < 0.40) {
        setRecommendation('ℹ️ La utilización (ρ) es baja (<40%). El sistema podría tener servidores ociosos, considere optimizar turnos.');
      } else {
        setRecommendation('✅ El sistema opera en un estado de equilibrio adecuado.');
      }
    }
  }, [studyId]);

  const handleCalculatePn = () => {
    if (!model) return;
    const n = parseInt(nInput);
    if (isNaN(n) || n < 0) return;
    const p = calculatePn(
      model.lambda_calculated, 
      model.mu_calculated, 
      model.servers_count, 
      n, 
      model.result_P0
    );
    setPnResult(p);
  };

  const generatePDF = async () => {
    if (!model || !study) return;
    
    const html = `
      <html>
        <head>
          <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .section { margin-bottom: 30px; }
            .card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #3498db; color: white; }
            .recommendation { background: #e8f4f8; border-left: 5px solid #3498db; padding: 15px; margin-top: 20px;}
          </style>
        </head>
        <body>
          <h1>Informe de Simulación de Colas</h1>
          
          <div class="section">
            <h2>Contexto del Estudio</h2>
            <p><strong>Título:</strong> ${study.title}</p>
            <p><strong>Descripción:</strong> ${study.context_description || 'N/A'}</p>
            <p><strong>Fecha:</strong> ${new Date(study.created_at).toLocaleString()}</p>
          </div>

          <div class="card">
            <h2>Configuración del Modelo</h2>
            <p><strong>Tipo:</strong> ${model.type === 'MM1' ? 'M/M/1' : 'M/M/S'}</p>
            <p><strong>Servidores (S):</strong> ${model.servers_count}</p>
            <p><strong>Tasa de Llegada (λ):</strong> ${model.lambda_calculated} clientes/tiempo</p>
            <p><strong>Tasa de Servicio (μ):</strong> ${model.mu_calculated} clientes/tiempo</p>
          </div>

          <div class="section">
            <h2>Resultados Matemáticos</h2>
            <table>
              <tr><th>Métrica</th><th>Valor</th><th>Interpretación</th></tr>
              <tr><td>ρ (Utilización)</td><td>${(model.result_Rho * 100).toFixed(2)}%</td><td>Ocupación del sistema</td></tr>
              <tr><td>L</td><td>${model.result_L.toFixed(4)}</td><td>Clientes prom. en el sistema</td></tr>
              <tr><td>Lq</td><td>${model.result_Lq.toFixed(4)}</td><td>Clientes prom. en la cola</td></tr>
              <tr><td>W</td><td>${model.result_W.toFixed(4)}</td><td>Tiempo prom. en el sistema</td></tr>
              <tr><td>Wq</td><td>${model.result_Wq.toFixed(4)}</td><td>Tiempo prom. en la cola</td></tr>
              <tr><td>P0</td><td>${(model.result_P0 * 100).toFixed(2)}%</td><td>Prob. de sistema vacío</td></tr>
            </table>
          </div>

          <div class="recommendation">
            <h2>Análisis y Propuesta de Mejora</h2>
            <p><strong>Sugerencia Automática:</strong><br/>${recommendation}</p>
            <p><strong>Anotaciones Adicionales:</strong><br/>${customNotes || 'Sin anotaciones adicionales.'}</p>
          </div>
        </body>
      </html>
    `;

    try {
      const { uri } = await Print.printToFileAsync({ html });
      await Sharing.shareAsync(uri);
    } catch (error) {
      Alert.alert('Error', 'No se pudo generar el PDF');
    }
  };

  if (!model) return <View style={styles.container}><Text>Cargando...</Text></View>;

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Surface style={styles.surface} elevation={1}>
        <Text variant="headlineSmall" style={{ marginBottom: 16 }}>{study?.title}</Text>
        
        <Card style={styles.card}>
          <Card.Content>
            <Text variant="titleMedium" style={{ marginBottom: 10 }}>Fórmulas Empleadas</Text>
            {model.type === 'MM1' ? (
              <Text variant="bodyLarge" style={styles.math}>L = λ / (μ - λ)</Text>
            ) : (
              <Text variant="bodyLarge" style={styles.math}>ρ = λ / (s · μ)</Text>
            )}
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Text variant="titleMedium" style={{ marginBottom: 10 }}>Métricas Estándar</Text>
            <Text variant="bodyLarge">ρ: {(model.result_Rho * 100).toFixed(2)}%</Text>
            <Text variant="bodyLarge">L: {model.result_L.toFixed(4)}</Text>
            <Text variant="bodyLarge">Lq: {model.result_Lq.toFixed(4)}</Text>
            <Text variant="bodyLarge">W: {model.result_W.toFixed(4)}</Text>
            <Text variant="bodyLarge">Wq: {model.result_Wq.toFixed(4)}</Text>
            <Text variant="bodyLarge">P0: {(model.result_P0 * 100).toFixed(2)}%</Text>
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Content>
            <Text variant="titleMedium" style={{ marginBottom: 10 }}>Probabilidad Pn</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <TextInput
                label="Valor de n"
                value={nInput}
                onChangeText={setNInput}
                keyboardType="numeric"
                mode="outlined"
                style={{ flex: 1, marginRight: 10 }}
              />
              <Button mode="contained-tonal" onPress={handleCalculatePn}>Calcular</Button>
            </View>
            {pnResult !== null && (
              <Text variant="bodyLarge" style={{ marginTop: 10 }}>
                P({nInput}) = {(pnResult * 100).toFixed(4)}%
              </Text>
            )}
          </Card.Content>
        </Card>

        <Card style={[styles.card, { backgroundColor: theme.colors.tertiaryContainer }]}>
          <Card.Content>
            <Text variant="titleMedium" style={{ color: theme.colors.onTertiaryContainer }}>Análisis y Mejora</Text>
            <Text style={{ marginTop: 8, color: theme.colors.onTertiaryContainer }}>{recommendation}</Text>
            
            <TextInput
              label="Notas Adicionales"
              value={customNotes}
              onChangeText={setCustomNotes}
              mode="outlined"
              multiline
              numberOfLines={3}
              style={{ marginTop: 12, backgroundColor: theme.colors.surface }}
            />
          </Card.Content>
        </Card>

        <Button mode="contained" style={styles.button} icon="file-pdf-box" onPress={generatePDF}>
          Generar Informe PDF
        </Button>
      </Surface>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  surface: { padding: 16, margin: 16, borderRadius: 12 },
  card: { marginBottom: 16 },
  button: { marginTop: 8, marginBottom: 20 },
  math: { marginVertical: 8, fontFamily: 'monospace', fontSize: 18 }
});
