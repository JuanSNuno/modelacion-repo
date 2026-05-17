import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ScrollView, Alert, TouchableOpacity, SafeAreaView, Platform, StatusBar, TextInput as RNTextInput } from 'react-native';
import { Text, useTheme, IconButton } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { calculatePn } from '../math/queueEngine';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system/legacy';
import { MaterialIcons } from '@expo/vector-icons';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Results'>;
  route: RouteProp<RootStackParamList, 'Results'>;
};

export default function ResultsScreen({ navigation, route }: Props) {
  const theme = useTheme();
  const { studyId } = route.params;

  const [model, setModel] = useState<any>(null);
  const [study, setStudy] = useState<any>(null);
  const [nInput, setNInput] = useState('3');
  const [pnResult, setPnResult] = useState<number | null>(null);
  const [userConclusions, setUserConclusions] = useState('');

  useEffect(() => {
    const studyData = db.getFirstSync('SELECT * FROM studies WHERE id = ?', [studyId]);
    const modelData = db.getFirstSync('SELECT * FROM queue_models WHERE study_id = ?', [studyId]);
    setStudy(studyData);
    setModel(modelData);

    if (modelData) {
      // Pre-calculate Pn for n=3 initially if wanted
      const p = calculatePn(
        (modelData as any).lambda_calculated, 
        (modelData as any).mu_calculated, 
        (modelData as any).servers_count, 
        3, 
        (modelData as any).result_P0
      );
      setPnResult(p);
    }
  }, [studyId]);

  const handleCalculatePn = (text: string) => {
    setNInput(text);
    if (!model) return;
    const n = parseInt(text);
    if (isNaN(n) || n < 0) {
      setPnResult(null);
      return;
    }
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
    
    let recommendation = '';
    if (model.result_Rho > 0.85) {
      recommendation = '⚠️ La utilización (ρ) es muy alta (>85%). Se recomienda agregar un servidor o reducir los tiempos de atención para evitar cuellos de botella.';
    } else if (model.result_Rho < 0.40) {
      recommendation = 'ℹ️ La utilización (ρ) es baja (<40%). El sistema podría tener servidores ociosos, considere optimizar turnos.';
    } else {
      recommendation = '✅ El sistema opera en un estado de equilibrio adecuado.';
    }

    const conclusionsHtml = userConclusions.trim() ? `
      <div class="section">
        <h2>Conclusiones del Usuario</h2>
        <div class="card" style="white-space: pre-wrap;">${userConclusions}</div>
      </div>
    ` : '';

    const html = `
      <html>
        <head>
          <style>
            @page { margin: 20mm; }
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 0; margin: 0; color: #333; }
            h1 { color: #000666; border-bottom: 2px solid #000666; padding-bottom: 10px; }
            .section { margin-bottom: 30px; page-break-inside: avoid; }
            .card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e2e2; page-break-inside: avoid; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; page-break-inside: avoid; }
            tr { page-break-inside: avoid; page-break-after: auto; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #000666; color: white; }
            .recommendation { background: #e8f4f8; border-left: 5px solid #000666; padding: 15px; margin-top: 20px; page-break-inside: avoid; }
          </style>
        </head>
        <body>
          <h1>Informe de Simulación de Colas - GasApp</h1>
          
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
            <p><strong>Tasa de Llegada (λ):</strong> ${model.lambda_calculated} veh/h</p>
            <p><strong>Tasa de Servicio (μ):</strong> ${model.mu_calculated} veh/h</p>
          </div>

          <div class="section">
            <h2>Resultados Matemáticos</h2>
            <table>
              <tr><th>Métrica</th><th>Valor</th><th>Interpretación</th></tr>
              <tr><td>ρ (Utilización)</td><td>${(model.result_Rho * 100).toFixed(2)}%</td><td>Ocupación del sistema</td></tr>
              <tr><td>L</td><td>${model.result_L.toFixed(4)}</td><td>Clientes prom. en el sistema</td></tr>
              <tr><td>Lq</td><td>${model.result_Lq.toFixed(4)}</td><td>Clientes prom. en la cola</td></tr>
              <tr><td>W</td><td>${model.result_W.toFixed(4)}</td><td>Tiempo prom. en el sistema (h)</td></tr>
              <tr><td>Wq</td><td>${model.result_Wq.toFixed(4)}</td><td>Tiempo prom. en la cola (h)</td></tr>
              <tr><td>P0</td><td>${(model.result_P0 * 100).toFixed(2)}%</td><td>Prob. de sistema vacío</td></tr>
            </table>
          </div>

          <div class="recommendation">
            <h2>Análisis Automático</h2>
            <p>${recommendation}</p>
          </div>
          
          ${conclusionsHtml}
        </body>
      </html>
    `;

    try {
      const { uri } = await Print.printToFileAsync({ html });
      
      // Sanitizar el título para usarlo como nombre de archivo
      const safeTitle = study.title ? study.title.replace(/[^a-zA-Z0-9_\-]/g, '_') : 'Resultados';
      const newUri = `${FileSystem.documentDirectory}${safeTitle}.pdf`;
      
      await FileSystem.moveAsync({
        from: uri,
        to: newUri,
      });
      
      await Sharing.shareAsync(newUri);
    } catch (error: any) {
      console.error(error);
      Alert.alert('Error', `No se pudo generar el PDF: ${error.message || error}`);
    }
  };

  if (!model || !study) {
    return (
      <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background, justifyContent: 'center', alignItems: 'center' }]}>
        <Text>Cargando resultados...</Text>
      </SafeAreaView>
    );
  }

  const rhoPercent = model.result_Rho * 100;
  const isHighUtil = rhoPercent > 85;

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      {/* Top Header */}
      <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
        <IconButton 
          icon="arrow-left" 
          iconColor={theme.colors.primary} 
          onPress={() => navigation.goBack()}
        />
        <Text variant="titleLarge" style={{ fontWeight: '700', color: theme.colors.primary, letterSpacing: -0.5 }}>
          GasApp
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        
        {/* Header Section */}
        <View style={styles.titleSection}>
          <Text variant="headlineSmall" style={{ fontWeight: '600', color: theme.colors.primary }}>
            Resultados del Análisis
          </Text>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginTop: 4 }}>
            Modelo de colas {model.type === 'MM1' ? 'M/M/1' : 'M/M/S'} - {study.title}
          </Text>
        </View>

        {/* KPI Grid */}
        <View style={styles.kpiGrid}>
          {/* Rho Card - Full Width */}
          <View style={[styles.card, styles.cardFull, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant, shadowColor: theme.colors.primary }]}>
            <View style={styles.bgIcon}>
              <MaterialIcons name="analytics" size={64} color={theme.colors.onSurfaceVariant} style={{ opacity: 0.1 }} />
            </View>
            <View style={styles.rhoHeader}>
              <MaterialIcons name="speed" size={18} color={theme.colors.onSurfaceVariant} />
              <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginLeft: 8 }}>
                ρ (Utilización)
              </Text>
            </View>
            <View style={styles.rhoValues}>
              <Text variant="displaySmall" style={{ fontWeight: '700', color: isHighUtil ? theme.colors.error : theme.colors.secondary, marginRight: 12 }}>
                {model.result_Rho.toFixed(2)}
              </Text>
              <Text variant="titleMedium" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 4, fontFamily: 'monospace' }}>
                {rhoPercent.toFixed(0)}%
              </Text>
            </View>
            <View style={[styles.progressBarBg, { backgroundColor: theme.colors.surfaceVariant }]}>
              <View style={[styles.progressBarFill, { backgroundColor: isHighUtil ? theme.colors.error : theme.colors.secondary, width: `${Math.min(rhoPercent, 100)}%` as any }]} />
            </View>
          </View>

          {/* KPI Small Cards */}
          <View style={[styles.card, styles.cardSmall, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
            <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 4 }}>L (Vehículos en sistema)</Text>
            <Text variant="headlineSmall" style={{ fontWeight: '700', color: theme.colors.primary }}>{model.result_L.toFixed(2)}</Text>
          </View>

          <View style={[styles.card, styles.cardSmall, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
            <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 4 }}>Lq (Vehículos en cola)</Text>
            <Text variant="headlineSmall" style={{ fontWeight: '700', color: theme.colors.primary }}>{model.result_Lq.toFixed(2)}</Text>
          </View>

          <View style={[styles.card, styles.cardSmall, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
            <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 4 }}>W (Tiempo en sistema)</Text>
            <View style={{ flexDirection: 'row', alignItems: 'baseline' }}>
              <Text variant="headlineSmall" style={{ fontWeight: '700', color: theme.colors.primary, marginRight: 4 }}>{(model.result_W * 60).toFixed(1)}</Text>
              <Text variant="labelMedium" style={{ color: theme.colors.onSurfaceVariant, fontFamily: 'monospace' }}>min</Text>
            </View>
          </View>

          <View style={[styles.card, styles.cardSmall, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
            <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 4 }}>Wq (Tiempo en cola)</Text>
            <View style={{ flexDirection: 'row', alignItems: 'baseline' }}>
              <Text variant="headlineSmall" style={{ fontWeight: '700', color: theme.colors.primary, marginRight: 4 }}>{(model.result_Wq * 60).toFixed(1)}</Text>
              <Text variant="labelMedium" style={{ color: theme.colors.onSurfaceVariant, fontFamily: 'monospace' }}>min</Text>
            </View>
          </View>
        </View>

        {/* Probability Calculator */}
        <View style={[styles.probCard, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
          <View style={[styles.probGlow, { backgroundColor: (theme.colors as any).primaryFixedDim }]} />
          
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 16 }}>
            <MaterialIcons name="calculate" size={24} color={theme.colors.primary} />
            <Text variant="titleMedium" style={{ fontWeight: '600', color: theme.colors.primary, marginLeft: 8 }}>
              Calculadora de Probabilidad Pn
            </Text>
          </View>
          
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 16 }}>
            Calcule la probabilidad de que haya exactamente 'n' clientes en el sistema.
          </Text>

          <View style={[styles.probCalculator, { backgroundColor: (theme.colors as any).surfaceContainer }]}>
            <View style={styles.probInputWrapper}>
              <Text style={[styles.probLabel, { color: theme.colors.primary, backgroundColor: (theme.colors as any).surfaceContainer }]}>
                Clientes (n)
              </Text>
              <RNTextInput
                style={[styles.probInput, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outline, color: theme.colors.onSurface }]}
                keyboardType="numeric"
                value={nInput}
                onChangeText={handleCalculatePn}
              />
            </View>
            
            <View style={styles.probArrow}>
              <MaterialIcons name="arrow-forward" size={24} color={theme.colors.outlineVariant} />
            </View>

            <View style={[styles.probResult, { backgroundColor: theme.colors.secondaryContainer, borderColor: (theme.colors as any).secondaryFixed }]}>
              <Text variant="bodyMedium" style={{ color: theme.colors.onSecondaryContainer }}>
                Probabilidad (P{nInput || 'n'})
              </Text>
              <Text variant="headlineSmall" style={{ fontWeight: '700', color: (theme.colors as any).onSecondaryFixed, marginTop: 4, fontFamily: 'monospace' }}>
                Pn = {pnResult !== null ? (pnResult * 100).toFixed(2) + '%' : '--'}
              </Text>
            </View>
          </View>
        </View>

        {/* User Conclusions */}
        <View style={styles.conclusionsContainer}>
          <Text variant="titleMedium" style={{ fontWeight: '600', color: theme.colors.primary, marginBottom: 8 }}>
            Mis Conclusiones
          </Text>
          <RNTextInput
            style={[styles.conclusionsInput, { 
              backgroundColor: (theme.colors as any).surfaceContainerLowest, 
              borderColor: theme.colors.outline, 
              color: theme.colors.onSurface 
            }]}
            multiline
            numberOfLines={4}
            placeholder="Escribe tus conclusiones aquí para incluirlas en el PDF..."
            placeholderTextColor={theme.colors.onSurfaceVariant}
            value={userConclusions}
            onChangeText={setUserConclusions}
            textAlignVertical="top"
          />
        </View>

        {/* Bottom Actions */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity 
            style={[styles.actionBtn, { borderColor: theme.colors.outline, marginBottom: 16 }]}
            onPress={() => navigation.navigate('Formulas', { studyId })}
            activeOpacity={0.7}
          >
            <MaterialIcons name="functions" size={20} color={theme.colors.primary} style={{ marginRight: 8 }} />
            <Text variant="titleMedium" style={{ color: theme.colors.primary, fontWeight: '600' }}>Ver Fórmulas</Text>
          </TouchableOpacity>

          <View style={styles.actions}>
            <TouchableOpacity 
              style={[styles.actionBtn, { borderColor: theme.colors.outline }]}
              onPress={generatePDF}
              activeOpacity={0.7}
            >
              <MaterialIcons name="picture-as-pdf" size={20} color={theme.colors.primary} style={{ marginRight: 8 }} />
              <Text variant="titleMedium" style={{ color: theme.colors.primary, fontWeight: '600' }}>Exportar PDF</Text>
            </TouchableOpacity>
          </View>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  actionsContainer: {
    flexDirection: 'column',
  },
  safeArea: {
    flex: 1,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  header: {
    height: 64,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    zIndex: 50,
  },
  content: {
    paddingHorizontal: 16,
    paddingBottom: 40,
  },
  titleSection: {
    marginTop: 24,
    marginBottom: 24,
  },
  kpiGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
    marginBottom: 12,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 1,
  },
  cardFull: {
    width: '100%',
    position: 'relative',
    overflow: 'hidden',
  },
  cardSmall: {
    width: '48%',
  },
  bgIcon: {
    position: 'absolute',
    top: 16,
    right: 16,
  },
  rhoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  rhoValues: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  progressBarBg: {
    height: 6,
    borderRadius: 3,
    marginTop: 16,
    width: '100%',
  },
  progressBarFill: {
    height: 6,
    borderRadius: 3,
  },
  probCard: {
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
    marginBottom: 24,
    position: 'relative',
    overflow: 'hidden',
  },
  probGlow: {
    position: 'absolute',
    right: -20,
    top: -20,
    width: 100,
    height: 100,
    borderRadius: 50,
    opacity: 0.2,
  },
  probCalculator: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 8,
    padding: 16,
  },
  probInputWrapper: {
    flex: 1,
    position: 'relative',
  },
  probLabel: {
    position: 'absolute',
    top: -8,
    left: 8,
    fontSize: 10,
    paddingHorizontal: 4,
    zIndex: 10,
  },
  probInput: {
    height: 48,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
    fontFamily: 'monospace',
  },
  probArrow: {
    paddingHorizontal: 16,
  },
  probResult: {
    flex: 2,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
  },
  conclusionsContainer: {
    marginBottom: 24,
  },
  conclusionsInput: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 16,
    fontSize: 16,
    minHeight: 120,
  },
  actions: {
    flexDirection: 'row',
    gap: 16,
  },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 48,
    borderRadius: 24,
    borderWidth: 1,
  },
});
