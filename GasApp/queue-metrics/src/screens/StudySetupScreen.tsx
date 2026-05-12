import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, SafeAreaView, Platform, StatusBar, TextInput as RNTextInput } from 'react-native';
import { Text, useTheme, IconButton } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { MaterialIcons } from '@expo/vector-icons';

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

      <View style={styles.content}>
        <View style={styles.titleSection}>
          <Text variant="headlineSmall" style={{ fontWeight: '600', color: theme.colors.onSurface }}>
            Nuevo Estudio
          </Text>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginTop: 8 }}>
            Configura los parámetros iniciales para el análisis de la estación.
          </Text>
        </View>

        <View style={[styles.formContainer, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
          
          <View style={styles.inputGroup}>
            <Text variant="bodyMedium" style={[styles.label, { color: theme.colors.onSurface }]}>
              Título del Estudio
            </Text>
            <View style={styles.inputWrapper}>
              <RNTextInput
                style={[styles.input, { 
                  backgroundColor: theme.colors.surface, 
                  borderColor: theme.colors.outlineVariant,
                  color: theme.colors.onSurface 
                }]}
                placeholder="Ej. Análisis Mañana Lunes"
                placeholderTextColor={theme.colors.outlineVariant}
                value={title}
                onChangeText={setTitle}
              />
              <MaterialIcons name="edit" size={20} color={theme.colors.outlineVariant} style={styles.inputIcon} />
            </View>
          </View>

          <View style={[styles.inputGroup, { flex: 1, marginTop: 24 }]}>
            <Text variant="bodyMedium" style={[styles.label, { color: theme.colors.onSurface }]}>
              Descripción
            </Text>
            <RNTextInput
              style={[styles.textArea, { 
                backgroundColor: theme.colors.surface, 
                borderColor: theme.colors.outlineVariant,
                color: theme.colors.onSurface 
              }]}
              placeholder="Ej. Medición de tiempos en hora pico para el modelo M/M/c..."
              placeholderTextColor={theme.colors.outlineVariant}
              value={description}
              onChangeText={setDescription}
              multiline
              numberOfLines={5}
              textAlignVertical="top"
            />
          </View>

        </View>

        <TouchableOpacity 
          style={[styles.button, { backgroundColor: title ? theme.colors.primary : theme.colors.surfaceVariant }]}
          onPress={handleCreate}
          disabled={!title}
          activeOpacity={0.8}
        >
          <Text variant="titleMedium" style={{ color: title ? theme.colors.onPrimary : theme.colors.onSurfaceVariant, fontWeight: '600', marginRight: 8 }}>
            Crear y Continuar
          </Text>
          <MaterialIcons name="arrow-forward" size={20} color={title ? theme.colors.onPrimary : theme.colors.onSurfaceVariant} />
        </TouchableOpacity>
      </View>
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
    flex: 1,
    paddingHorizontal: 16,
    paddingBottom: 24,
  },
  titleSection: {
    marginTop: 24,
    marginBottom: 24,
  },
  formContainer: {
    flex: 1,
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  inputGroup: {
    flexDirection: 'column',
  },
  label: {
    fontWeight: '600',
    marginBottom: 8,
  },
  inputWrapper: {
    position: 'relative',
    justifyContent: 'center',
  },
  input: {
    height: 48,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingRight: 40, // Space for icon
    fontSize: 16,
  },
  inputIcon: {
    position: 'absolute',
    right: 12,
  },
  textArea: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    minHeight: 120,
  },
  button: {
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
  },
});
