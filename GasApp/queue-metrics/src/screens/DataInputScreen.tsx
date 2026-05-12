import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert, TouchableOpacity, SafeAreaView, Platform, StatusBar, TextInput as RNTextInput } from 'react-native';
import { Text, useTheme, IconButton } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { calculateMM1, calculateMMS } from '../math/queueEngine';
import { MaterialIcons } from '@expo/vector-icons';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'DataInput'>;
  route: RouteProp<RootStackParamList, 'DataInput'>;
};

export default function DataInputScreen({ navigation, route }: Props) {
  const theme = useTheme();
  const { studyId } = route.params;

  const [modelType, setModelType] = useState('MM1');
  const [lambda, setLambda] = useState('45.5');
  const [mu, setMu] = useState('60.0');
  const [servers, setServers] = useState('1');

  const handleSave = () => {
    if (!lambda || !mu) return;

    try {
      const lam = parseFloat(lambda);
      const m = parseFloat(mu);
      const s = parseInt(servers);

      let results;
      if (modelType === 'MM1') {
        results = calculateMM1(lam, m);
      } else {
        if (s < 1) throw new Error('Debe haber al menos 1 servidor');
        results = calculateMMS(lam, m, s);
      }

      db.runSync(
        `INSERT INTO queue_models 
          (id, study_id, type, servers_count, lambda_calculated, mu_calculated, result_L, result_Lq, result_W, result_Wq, result_P0, result_Rho) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          Date.now().toString(), studyId, modelType, s, lam, m, 
          results.L, results.Lq, results.W, results.Wq, results.P0, results.rho
        ]
      );

      navigation.replace('Results', { studyId });
    } catch (error: any) {
      Alert.alert('Error de Cálculo', error.message);
    }
  };

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      {/* Top Header */}
      <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
        <IconButton 
          icon="arrow-left" 
          iconColor={theme.colors.onSurfaceVariant} 
          onPress={() => navigation.goBack()}
        />
        <Text variant="titleLarge" style={{ fontWeight: '700', color: theme.colors.primary, letterSpacing: -0.5 }}>
          GasApp
        </Text>
        <IconButton icon="dots-vertical" iconColor={theme.colors.onSurfaceVariant} />
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.titleSection}>
          <Text variant="headlineSmall" style={{ fontWeight: '600', color: theme.colors.onSurface }}>
            Parámetros del Modelo
          </Text>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginTop: 8 }}>
            Configure los parámetros de entrada para la simulación de colas en la estación de servicio.
          </Text>
        </View>

        {/* Analytical Input Card */}
        <View style={[styles.card, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
          {/* Subtle Depth Cue Glow (Approximated with a View) */}
          <View style={[styles.glow, { backgroundColor: `${theme.colors.primary}0D` }]} />
          
          <View style={styles.section}>
            <Text variant="labelLarge" style={{ color: theme.colors.onSurfaceVariant, marginBottom: 8, fontFamily: 'monospace' }}>
              Modelo de Simulación
            </Text>
            <View style={[styles.segmentedControl, { backgroundColor: (theme.colors as any).surfaceContainerLow, borderColor: 'rgba(198, 197, 212, 0.2)' }]}>
              <TouchableOpacity 
                style={[styles.segmentBtn, modelType === 'MM1' ? { backgroundColor: theme.colors.primary } : {}]}
                onPress={() => setModelType('MM1')}
                activeOpacity={0.8}
              >
                <Text style={[styles.segmentText, { color: modelType === 'MM1' ? theme.colors.onPrimary : theme.colors.onSurfaceVariant }]}>
                  M/M/1
                </Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.segmentBtn, modelType === 'MMS' ? { backgroundColor: theme.colors.primary } : {}]}
                onPress={() => setModelType('MMS')}
                activeOpacity={0.8}
              >
                <Text style={[styles.segmentText, { color: modelType === 'MMS' ? theme.colors.onPrimary : theme.colors.onSurfaceVariant }]}>
                  M/M/S
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={[styles.divider, { borderBottomColor: 'rgba(198, 197, 212, 0.3)' }]} />

          <View style={styles.grid}>
            {/* Lambda Input */}
            <View style={styles.inputContainer}>
              <Text style={[styles.floatingLabel, { color: theme.colors.primary, backgroundColor: (theme.colors as any).surfaceContainerLowest }]}>
                Tasa de Llegada (λ)
              </Text>
              <View style={[styles.inputWrapper, { borderColor: theme.colors.primary, borderWidth: 2 }]}>
                <RNTextInput
                  style={[styles.inputNumeric, { color: theme.colors.onSurface }]}
                  keyboardType="numeric"
                  value={lambda}
                  onChangeText={setLambda}
                  placeholder="0.00"
                  placeholderTextColor={theme.colors.outlineVariant}
                />
                <Text style={[styles.unitText, { color: theme.colors.onSurfaceVariant }]}>veh/h</Text>
              </View>
              <Text style={[styles.helperText, { color: theme.colors.onSurfaceVariant }]}>Vehículos llegando por hora.</Text>
            </View>

            {/* Mu Input */}
            <View style={styles.inputContainer}>
              <Text style={[styles.floatingLabel, { color: theme.colors.onSurfaceVariant, backgroundColor: (theme.colors as any).surfaceContainerLowest }]}>
                Tasa de Servicio (μ)
              </Text>
              <View style={[styles.inputWrapper, { borderColor: theme.colors.outlineVariant, borderWidth: 1 }]}>
                <RNTextInput
                  style={[styles.inputNumeric, { color: theme.colors.onSurface }]}
                  keyboardType="numeric"
                  value={mu}
                  onChangeText={setMu}
                  placeholder="0.00"
                  placeholderTextColor={theme.colors.outlineVariant}
                />
                <Text style={[styles.unitText, { color: theme.colors.onSurfaceVariant }]}>veh/h</Text>
              </View>
              <Text style={[styles.helperText, { color: theme.colors.onSurfaceVariant }]}>Vehículos atendidos por hora por servidor.</Text>
            </View>

            {/* Servers Input */}
            <View style={[styles.inputContainer, modelType === 'MM1' ? { opacity: 0.5 } : {}]}>
              <Text style={[styles.floatingLabel, { color: theme.colors.onSurfaceVariant, backgroundColor: (theme.colors as any).surfaceContainerLowest }]}>
                Número de Servidores (S)
              </Text>
              <View style={[styles.inputWrapper, { borderColor: theme.colors.outlineVariant, borderWidth: 1, backgroundColor: modelType === 'MM1' ? (theme.colors as any).surfaceContainerLow : 'transparent' }]}>
                <RNTextInput
                  style={[styles.inputNumeric, { color: theme.colors.onSurface }]}
                  keyboardType="numeric"
                  value={servers}
                  onChangeText={setServers}
                  placeholder="3"
                  editable={modelType === 'MMS'}
                  placeholderTextColor={theme.colors.outlineVariant}
                />
              </View>
              <Text style={[styles.helperText, { color: theme.colors.onSurfaceVariant }]}>
                {modelType === 'MM1' ? 'Deshabilitado para modelo M/M/1.' : 'Cantidad de servidores paralelos.'}
              </Text>
            </View>
          </View>
        </View>

        {/* Action Area */}
        <View style={styles.actionArea}>
          <TouchableOpacity 
            style={[styles.actionBtn, { backgroundColor: theme.colors.primary }]}
            onPress={handleSave}
            activeOpacity={0.9}
          >
            <MaterialIcons name="calculate" size={24} color={theme.colors.onPrimary} style={{ marginRight: 8 }} />
            <Text variant="titleLarge" style={{ color: theme.colors.onPrimary, fontWeight: '600' }}>
              Calcular Métricas
            </Text>
          </TouchableOpacity>
          
          <View style={styles.infoWrapper}>
            <MaterialIcons name="info" size={16} color={theme.colors.onSurfaceVariant} />
            <Text style={[styles.infoText, { color: theme.colors.onSurfaceVariant }]}>
              Asegúrese de que μ {'>'} λ para evitar colas infinitas.
            </Text>
          </View>
        </View>

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
  card: {
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  glow: {
    position: 'absolute',
    top: -40,
    right: -40,
    width: 128,
    height: 128,
    borderRadius: 64,
  },
  section: {
    marginBottom: 16,
  },
  segmentedControl: {
    flexDirection: 'row',
    padding: 4,
    borderRadius: 8,
    borderWidth: 1,
  },
  segmentBtn: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 6,
    alignItems: 'center',
  },
  segmentText: {
    fontWeight: '600',
    fontSize: 16,
  },
  divider: {
    borderBottomWidth: 1,
    marginVertical: 16,
  },
  grid: {
    gap: 24,
  },
  inputContainer: {
    position: 'relative',
    marginBottom: 8,
  },
  floatingLabel: {
    position: 'absolute',
    top: -10,
    left: 12,
    paddingHorizontal: 4,
    fontSize: 12,
    zIndex: 10,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 8,
    height: 52,
    paddingHorizontal: 16,
  },
  inputNumeric: {
    flex: 1,
    fontSize: 20,
    fontFamily: 'monospace',
    fontWeight: '600',
    height: '100%',
  },
  unitText: {
    fontSize: 14,
    marginLeft: 8,
  },
  helperText: {
    fontSize: 12,
    marginTop: 6,
    marginLeft: 4,
  },
  actionArea: {
    marginTop: 16,
  },
  actionBtn: {
    flexDirection: 'row',
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 16,
  },
  infoWrapper: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
  },
  infoText: {
    fontSize: 12,
  },
});
