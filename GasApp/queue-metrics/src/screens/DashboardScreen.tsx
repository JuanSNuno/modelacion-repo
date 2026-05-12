import React, { useEffect, useState } from 'react';
import { View, StyleSheet, FlatList, TouchableOpacity, SafeAreaView, Platform, StatusBar } from 'react-native';
import { Text, useTheme, IconButton } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';
import { MaterialIcons } from '@expo/vector-icons';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Dashboard'>;
};

type StudyWithModel = {
  id: string;
  title: string;
  created_at: string;
  modelType: string | null;
  rho: number | null;
};

export default function DashboardScreen({ navigation }: Props) {
  const theme = useTheme();
  const [studies, setStudies] = useState<StudyWithModel[]>([]);

  const loadStudies = () => {
    // Left join to get model data if available
    const result = db.getAllSync<any>(`
      SELECT s.id, s.title, s.created_at, q.type as modelType, q.result_Rho as rho 
      FROM studies s
      LEFT JOIN queue_models q ON s.id = q.study_id
      ORDER BY s.created_at DESC
    `);
    setStudies(result);
  };

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadStudies();
    });
    return unsubscribe;
  }, [navigation]);

  const renderItem = ({ item }: { item: StudyWithModel }) => {
    const isHighUtil = item.rho && item.rho > 0.85;
    const utilizationText = item.rho !== null ? `${(item.rho * 100).toFixed(0)}%` : 'N/A';
    const utilizationColor = isHighUtil ? theme.colors.error : (item.rho ? theme.colors.secondary : theme.colors.onSurface);
    const indicatorColor = isHighUtil ? theme.colors.error : (item.rho ? theme.colors.secondary : theme.colors.outlineVariant);

    const formattedDate = new Date(item.created_at).toLocaleDateString('es-ES', { 
      day: 'numeric', month: 'short', year: 'numeric' 
    });

    return (
      <TouchableOpacity 
        style={[styles.card, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: 'rgba(198, 197, 212, 0.3)' }]}
        onPress={() => navigation.navigate('Results', { studyId: item.id })}
        activeOpacity={0.8}
      >
        <View style={[styles.cardIndicator, { backgroundColor: indicatorColor }]} />
        <View style={styles.cardHeader}>
          <View style={styles.cardHeaderLeft}>
            <View style={[styles.iconContainer, { backgroundColor: `${theme.colors.primaryContainer}1A` }]}>
              <MaterialIcons name="local-gas-station" size={20} color={theme.colors.primary} />
            </View>
            <View>
              <Text variant="titleMedium" style={{ fontWeight: '600', color: theme.colors.onSurface }}>
                {item.title}
              </Text>
              <View style={styles.dateContainer}>
                <MaterialIcons name="calendar-today" size={12} color={theme.colors.onSurfaceVariant} />
                <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, marginLeft: 4 }}>
                  {formattedDate}
                </Text>
              </View>
            </View>
          </View>
          <IconButton icon="chevron-right" size={20} iconColor={theme.colors.onSurfaceVariant} />
        </View>

        <View style={[styles.cardBody, { borderTopColor: 'rgba(198, 197, 212, 0.2)' }]}>
          <View style={styles.metricCol}>
            <Text variant="bodySmall" style={{ color: theme.colors.outline }}>Modelo</Text>
            <Text variant="titleSmall" style={{ fontFamily: 'monospace', fontWeight: 'bold', color: theme.colors.onSurface }}>
              {item.modelType === 'MM1' ? 'M/M/1' : (item.modelType === 'MMS' ? 'M/M/S' : 'Pendiente')}
            </Text>
          </View>
          <View style={styles.metricCol}>
            <Text variant="bodySmall" style={{ color: theme.colors.outline }}>Utilización</Text>
            <Text variant="titleSmall" style={{ fontFamily: 'monospace', fontWeight: 'bold', color: utilizationColor }}>
              {utilizationText}
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      {/* Top Header */}
      <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
        <View style={styles.headerLeft}>
          <Text variant="headlineSmall" style={{ fontWeight: '700', color: theme.colors.primary, letterSpacing: -0.5 }}>
            GasApp
          </Text>
        </View>
        <IconButton icon="dots-vertical" iconColor={theme.colors.onSurfaceVariant} />
      </View>

      {/* Main Content */}
      <View style={styles.content}>
        <View style={styles.titleSection}>
          <Text variant="headlineSmall" style={{ fontWeight: '600', color: theme.colors.onSurface }}>
            Mis Estudios
          </Text>
          <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, marginTop: 4 }}>
            Análisis de flujo y colas en estaciones.
          </Text>
        </View>

        {studies.length === 0 ? (
          <View style={[styles.emptyState, { backgroundColor: (theme.colors as any).surfaceContainerLowest, borderColor: theme.colors.outlineVariant }]}>
            <View style={[styles.emptyIconBg, { backgroundColor: theme.colors.surfaceVariant }]}>
              <MaterialIcons name="analytics" size={48} color={theme.colors.outline} />
            </View>
            <Text variant="titleMedium" style={{ fontWeight: '600', color: theme.colors.onSurface, marginBottom: 8 }}>
              No tienes estudios aún
            </Text>
            <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant, textAlign: 'center' }}>
              Comienza creando uno nuevo para analizar el rendimiento de tu estación.
            </Text>
          </View>
        ) : (
          <FlatList
            data={studies}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.list}
            renderItem={renderItem}
            showsVerticalScrollIndicator={false}
          />
        )}
      </View>

      {/* FAB */}
      <TouchableOpacity 
        style={[styles.fab, { backgroundColor: theme.colors.primary }]}
        activeOpacity={0.9}
        onPress={() => navigation.navigate('StudySetup')}
      >
        <MaterialIcons name="add" size={28} color={theme.colors.onPrimary} />
      </TouchableOpacity>
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
    paddingHorizontal: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    zIndex: 50,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  content: {
    flex: 1,
    paddingHorizontal: 16,
  },
  titleSection: {
    marginTop: 24,
    marginBottom: 24,
  },
  list: {
    paddingBottom: 100,
  },
  card: {
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    marginBottom: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 1,
  },
  cardIndicator: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 4,
    opacity: 0.8,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    padding: 8,
    borderRadius: 8,
    marginRight: 12,
  },
  dateContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  cardBody: {
    flexDirection: 'row',
    paddingTop: 16,
    borderTopWidth: 1,
  },
  metricCol: {
    flex: 1,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 64,
    paddingHorizontal: 24,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderRadius: 16,
    marginTop: 32,
  },
  emptyIconBg: {
    padding: 24,
    borderRadius: 100,
    marginBottom: 24,
  },
  fab: {
    position: 'absolute',
    right: 24,
    bottom: 24,
    width: 64,
    height: 64,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    zIndex: 40,
  },
});
