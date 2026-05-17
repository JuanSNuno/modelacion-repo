import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ScrollView, SafeAreaView, Platform, StatusBar } from 'react-native';
import { Text, useTheme, IconButton, Card } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { MaterialIcons } from '@expo/vector-icons';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Formulas'>;
  route: RouteProp<RootStackParamList, 'Formulas'>;
};

export default function FormulasScreen({ navigation, route }: Props) {
  const theme = useTheme();
  const { studyId } = route.params;
  const [model, setModel] = useState<any>(null);

  useEffect(() => {
    const modelData = db.getFirstSync('SELECT * FROM queue_models WHERE study_id = ?', [studyId]);
    setModel(modelData);
  }, [studyId]);

  if (!model) {
    return (
      <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background, justifyContent: 'center', alignItems: 'center' }]}>
        <Text>Cargando datos del modelo...</Text>
      </SafeAreaView>
    );
  }

  const l = model.lambda_calculated;
  const m = model.mu_calculated;
  const s = model.servers_count;
  const type = model.type;

  const renderFormulaCard = (title: string, formula: string, substitution: string, result: string, resultLabel: string) => (
    <Card style={[styles.card, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]} mode="outlined">
      <Card.Content>
        <Text variant="titleMedium" style={{ color: theme.colors.primary, fontWeight: '600', marginBottom: 8 }}>{title}</Text>
        
        <View style={[styles.formulaBox, { backgroundColor: (theme.colors as any).surfaceContainer }]}>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, fontFamily: 'monospace' }}>
            {formula}
          </Text>
        </View>

        <View style={styles.substitutionBox}>
          <MaterialIcons name="subdirectory-arrow-right" size={16} color={theme.colors.outlineVariant} style={{ marginTop: 2, marginRight: 4 }} />
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, flex: 1 }}>
            {substitution}
          </Text>
        </View>

        <View style={[styles.resultBox, { backgroundColor: theme.colors.secondaryContainer, borderColor: (theme.colors as any).secondaryFixed }]}>
          <Text variant="titleMedium" style={{ color: (theme.colors as any).onSecondaryFixed, fontWeight: '700', fontFamily: 'monospace' }}>
            {resultLabel} = {result}
          </Text>
        </View>
      </Card.Content>
    </Card>
  );

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
        <IconButton 
          icon="arrow-left" 
          iconColor={theme.colors.primary} 
          onPress={() => navigation.goBack()}
        />
        <Text variant="titleLarge" style={{ fontWeight: '700', color: theme.colors.primary, letterSpacing: -0.5 }}>
          Detalle de Fórmulas
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.titleSection}>
          <Text variant="headlineSmall" style={{ fontWeight: '600', color: theme.colors.primary }}>
            Modelo {type === 'MM1' ? 'M/M/1' : 'M/M/s'}
          </Text>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginTop: 4 }}>
            λ (Llegadas) = {l.toFixed(4)} / hora{'\n'}
            μ (Servicio) = {m.toFixed(4)} / hora{'\n'}
            s (Servidores) = {s}
          </Text>
        </View>

        {type === 'MM1' ? (
          <>
            {renderFormulaCard(
              'Utilización (ρ)', 
              'ρ = λ / μ', 
              `ρ = ${l.toFixed(4)} / ${m.toFixed(4)}`, 
              model.result_Rho.toFixed(4), 
              'ρ'
            )}
            {renderFormulaCard(
              'Probabilidad de 0 clientes (P0)', 
              'P0 = 1 - ρ', 
              `P0 = 1 - ${model.result_Rho.toFixed(4)}`, 
              model.result_P0.toFixed(4), 
              'P0'
            )}
            {renderFormulaCard(
              'Clientes en el sistema (L)', 
              'L = λ / (μ - λ)', 
              `L = ${l.toFixed(4)} / (${m.toFixed(4)} - ${l.toFixed(4)})`, 
              model.result_L.toFixed(4), 
              'L'
            )}
            {renderFormulaCard(
              'Clientes en la cola (Lq)', 
              'Lq = ρ² / (1 - ρ)', 
              `Lq = ${model.result_Rho.toFixed(4)}² / (1 - ${model.result_Rho.toFixed(4)})`, 
              model.result_Lq.toFixed(4), 
              'Lq'
            )}
            {renderFormulaCard(
              'Tiempo en el sistema (W)', 
              'W = 1 / (μ - λ)', 
              `W = 1 / (${m.toFixed(4)} - ${l.toFixed(4)})`, 
              `${model.result_W.toFixed(4)} hrs`, 
              'W'
            )}
            {renderFormulaCard(
              'Tiempo en la cola (Wq)', 
              'Wq = ρ / (μ - λ)', 
              `Wq = ${model.result_Rho.toFixed(4)} / (${m.toFixed(4)} - ${l.toFixed(4)})`, 
              `${model.result_Wq.toFixed(4)} hrs`, 
              'Wq'
            )}
          </>
        ) : (
          <>
            {renderFormulaCard(
              'Utilización (ρ)', 
              'ρ = λ / (s * μ)', 
              `ρ = ${l.toFixed(4)} / (${s} * ${m.toFixed(4)})`, 
              model.result_Rho.toFixed(4), 
              'ρ'
            )}
            {renderFormulaCard(
              'Probabilidad de 0 clientes (P0)', 
              'P0 = [ Σ(λ/μ)^n / n! + ((λ/μ)^s / s!) * (1 / (1 - ρ)) ] ^ -1', 
              `Suma hasta n=${s-1}, luego fórmula para s=${s}`, 
              model.result_P0.toFixed(4), 
              'P0'
            )}
            {renderFormulaCard(
              'Clientes en la cola (Lq)', 
              'Lq = [ (λ/μ)^s * ρ / (s! * (1 - ρ)²) ] * P0', 
              `Lq = [ (${(l/m).toFixed(4)})^${s} * ${model.result_Rho.toFixed(4)} / (${s}! * (1 - ${model.result_Rho.toFixed(4)})²) ] * ${model.result_P0.toFixed(4)}`, 
              model.result_Lq.toFixed(4), 
              'Lq'
            )}
            {renderFormulaCard(
              'Tiempo en la cola (Wq)', 
              'Wq = Lq / λ', 
              `Wq = ${model.result_Lq.toFixed(4)} / ${l.toFixed(4)}`, 
              `${model.result_Wq.toFixed(4)} hrs`, 
              'Wq'
            )}
            {renderFormulaCard(
              'Tiempo en el sistema (W)', 
              'W = Wq + 1/μ', 
              `W = ${model.result_Wq.toFixed(4)} + 1/${m.toFixed(4)}`, 
              `${model.result_W.toFixed(4)} hrs`, 
              'W'
            )}
            {renderFormulaCard(
              'Clientes en el sistema (L)', 
              'L = λ * W', 
              `L = ${l.toFixed(4)} * ${model.result_W.toFixed(4)}`, 
              model.result_L.toFixed(4), 
              'L'
            )}
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  header: {
    height: 64,
    flexDirection: 'row',
    alignItems: 'center',
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
  card: {
    marginBottom: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  formulaBox: {
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  substitutionBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
    paddingHorizontal: 4,
  },
  resultBox: {
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
  }
});
