import React, { useEffect, useState } from 'react';
import { View, StyleSheet, FlatList } from 'react-native';
import { FAB, Card, Text, useTheme } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { db } from '../database/db';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Dashboard'>;
};

type Study = {
  id: string;
  title: string;
  created_at: string;
};

export default function DashboardScreen({ navigation }: Props) {
  const theme = useTheme();
  const [studies, setStudies] = useState<Study[]>([]);

  const loadStudies = () => {
    const result = db.getAllSync<Study>('SELECT * FROM studies ORDER BY created_at DESC');
    setStudies(result);
  };

  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadStudies();
    });
    return unsubscribe;
  }, [navigation]);

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {studies.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text variant="bodyLarge" style={{ color: theme.colors.onSurfaceVariant }}>
            No hay estudios creados. Presiona el botón + para empezar.
          </Text>
        </View>
      ) : (
        <FlatList
          data={studies}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <Card 
              style={styles.card} 
              onPress={() => navigation.navigate('Results', { studyId: item.id })}
            >
              <Card.Title title={item.title} subtitle={`Creado: ${item.created_at}`} />
            </Card>
          )}
        />
      )}

      <FAB
        icon="plus"
        style={[styles.fab, { backgroundColor: theme.colors.primaryContainer }]}
        color={theme.colors.onPrimaryContainer}
        onPress={() => navigation.navigate('StudySetup')}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  list: {
    padding: 16,
  },
  card: {
    marginBottom: 12,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
});
