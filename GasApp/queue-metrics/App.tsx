import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { PaperProvider, MD3LightTheme, MD3DarkTheme } from 'react-native-paper';
import { useColorScheme } from 'react-native';

import DashboardScreen from './src/screens/DashboardScreen';
import StudySetupScreen from './src/screens/StudySetupScreen';
import DataInputScreen from './src/screens/DataInputScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import { initDb } from './src/database/db';

export type RootStackParamList = {
  Dashboard: undefined;
  StudySetup: undefined;
  DataInput: { studyId: string };
  Results: { studyId: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

// Initialize database on startup
initDb();

export default function App() {
  const colorScheme = useColorScheme();
  const theme = colorScheme === 'dark' ? MD3DarkTheme : MD3LightTheme;

  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Stack.Navigator 
          initialRouteName="Dashboard"
          screenOptions={{
            headerStyle: {
              backgroundColor: theme.colors.surface,
            },
            headerTintColor: theme.colors.onSurface,
          }}
        >
          <Stack.Screen 
            name="Dashboard" 
            component={DashboardScreen} 
            options={{ title: 'QueueMetrics' }}
          />
          <Stack.Screen 
            name="StudySetup" 
            component={StudySetupScreen} 
            options={{ title: 'Nuevo Estudio' }}
          />
          <Stack.Screen 
            name="DataInput" 
            component={DataInputScreen} 
            options={{ title: 'Recolección de Datos' }}
          />
          <Stack.Screen 
            name="Results" 
            component={ResultsScreen} 
            options={{ title: 'Resultados y Análisis' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
