import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { TextInput, Button, useTheme, Text, Surface, SegmentedButtons } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { calculateMM1, calculateMMS } from '../math/queueEngine';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'DataInput'>;
  route: RouteProp<RootStackParamList, 'DataInput'>;
};

export default function DataInputScreen({ navigation, route }: Props) {
  const theme = useTheme();
  const { studyId } = route.params;

  const [modelType, setModelType] = useState('MM1');
  const [lambda, setLambda] = useState('');
  const [mu, setMu] = useState('');
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
    <ScrollView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Surface style={styles.surface} elevation={1}>
        <Text variant="titleLarge" style={styles.title}>Tipo de Modelo</Text>
        <SegmentedButtons
          value={modelType}
          onValueChange={setModelType}
          buttons={[
            { value: 'MM1', label: 'M/M/1' },
            { value: 'MMS', label: 'M/M/S' },
          ]}
          style={styles.segmented}
        />

        {modelType === 'MMS' && (
          <TextInput
            label="Número de Servidores (S)"
            value={servers}
            onChangeText={setServers}
            keyboardType="numeric"
            mode="outlined"
            style={styles.input}
          />
        )}

        <Text variant="titleLarge" style={styles.title}>Tasas (Clientes por unidad de tiempo)</Text>
        <TextInput
          label="Tasa de Llegada (λ)"
          value={lambda}
          onChangeText={setLambda}
          keyboardType="numeric"
          mode="outlined"
          style={styles.input}
          placeholder="Ej: 5"
        />

        <TextInput
          label="Tasa de Servicio (μ)"
          value={mu}
          onChangeText={setMu}
          keyboardType="numeric"
          mode="outlined"
          style={styles.input}
          placeholder="Ej: 8"
        />

        <Button mode="contained" onPress={handleSave} style={styles.button} disabled={!lambda || !mu}>
          Calcular y Ver Resultados
        </Button>
      </Surface>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  surface: {
    padding: 16,
    margin: 16,
    borderRadius: 12,
  },
  title: {
    marginBottom: 12,
    marginTop: 8,
  },
  segmented: {
    marginBottom: 20,
  },
  input: {
    marginBottom: 16,
  },
  button: {
    marginTop: 16,
  },
});
