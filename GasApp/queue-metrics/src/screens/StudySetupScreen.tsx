import React, { useState } from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { TextInput, Button, useTheme, Text, Surface } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'StudySetup'>;
};

export default function StudySetupScreen({ navigation }: Props) {
  const theme = useTheme();
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');

  const handleCreate = () => {
    if (!title) return;
    
    const id = Date.now().toString();
    db.runSync(
      'INSERT INTO studies (id, title, context_description) VALUES (?, ?, ?)',
      [id, title, description]
    );

    navigation.replace('DataInput', { studyId: id });
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Surface style={styles.surface} elevation={1}>
        <Text variant="headlineSmall" style={{ marginBottom: 16 }}>Configuración del Contexto</Text>
        
        <TextInput
          label="Título del Estudio"
          value={title}
          onChangeText={setTitle}
          mode="outlined"
          style={styles.input}
          placeholder="Ej: Cajero de Banco"
        />

        <TextInput
          label="Descripción (Opcional)"
          value={description}
          onChangeText={setDescription}
          mode="outlined"
          multiline
          numberOfLines={4}
          style={styles.input}
          placeholder="Describe el flujo de atención y variables observadas."
        />

        <Button 
          mode="contained" 
          onPress={handleCreate} 
          style={styles.button}
          disabled={!title}
        >
          Continuar a Recolección
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
  input: {
    marginBottom: 16,
  },
  button: {
    marginTop: 8,
  },
});
